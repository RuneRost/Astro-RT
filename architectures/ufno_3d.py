import jax
import jax.numpy as jnp
import equinox as eqx

class SpectralConv3d(eqx.Module):
    real_weights1: jax.Array
    imag_weights1: jax.Array
    real_weights2: jax.Array
    imag_weights2: jax.Array
    real_weights3: jax.Array
    imag_weights3: jax.Array
    real_weights4: jax.Array
    imag_weights4: jax.Array
    in_channels: int
    out_channels: int
    modes_x: int
    modes_y: int
    modes_z: int
    
    def __init__(self, in_channels, out_channels, modes_x, modes_y, modes_z, *, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
 
        scale = jnp.sqrt(1.0 / (in_channels + out_channels)) 

        keys = jax.random.split(key, 8)
        self.real_weights1 = jax.random.uniform(keys[0], (in_channels, out_channels, modes_x, modes_y, modes_z), minval=-scale, maxval=+scale)
        self.imag_weights1 = jax.random.uniform(keys[1], (in_channels, out_channels, modes_x, modes_y, modes_z), minval=-scale, maxval=+scale)  
        self.real_weights2 = jax.random.uniform(keys[2], (in_channels, out_channels, modes_x, modes_y, modes_z), minval=-scale, maxval=+scale)
        self.imag_weights2 = jax.random.uniform(keys[3], (in_channels, out_channels, modes_x, modes_y, modes_z), minval=-scale, maxval=+scale)  
        self.real_weights3 = jax.random.uniform(keys[4], (in_channels, out_channels, modes_x, modes_y, modes_z), minval=-scale, maxval=+scale)
        self.imag_weights3 = jax.random.uniform(keys[5], (in_channels, out_channels, modes_x, modes_y, modes_z), minval=-scale, maxval=+scale)  
        self.real_weights4 = jax.random.uniform(keys[6], (in_channels, out_channels, modes_x, modes_y, modes_z), minval=-scale, maxval=+scale)
        self.imag_weights4 = jax.random.uniform(keys[7], (in_channels, out_channels, modes_x, modes_y, modes_z), minval=-scale, maxval=+scale)  

    def complex_mult3d(self, x_hat, w):
        return jnp.einsum("iXYZ,ioXYZ->oXYZ", x_hat, w)  
    
    def __call__(self, x):
        channels, spatial_points_x, spatial_points_y, spatial_points_z = x.shape
        x_hat = jnp.fft.rfftn(x, axes=(1, 2, 3))  

        weights1 = self.real_weights1 + 1j * self.imag_weights1
        weights2 = self.real_weights2 + 1j * self.imag_weights2
        weights3 = self.real_weights3 + 1j * self.imag_weights3
        weights4 = self.real_weights4 + 1j * self.imag_weights4

        out_hat = jnp.zeros((self.out_channels, *x_hat.shape[1:]),dtype=x_hat.dtype) 
        out_hat = out_hat.at[:, :self.modes_x, :self.modes_y, :self.modes_z].set(self.complex_mult3d(x_hat[:, :self.modes_x, :self.modes_y, :self.modes_z], weights1))
        out_hat = out_hat.at[:, -self.modes_x:, :self.modes_y, :self.modes_z].add(self.complex_mult3d(x_hat[:, -self.modes_x:, :self.modes_y, :self.modes_z], weights2))
        out_hat = out_hat.at[:, :self.modes_x, -self.modes_y:, :self.modes_z].add(self.complex_mult3d(x_hat[:, :self.modes_x, -self.modes_y:, :self.modes_z], weights3))
        out_hat = out_hat.at[:, -self.modes_x:, -self.modes_y:, :self.modes_z].add(self.complex_mult3d(x_hat[:, -self.modes_x:, -self.modes_y:, :self.modes_z], weights4))
 
        out = jnp.fft.irfftn(out_hat, s=[spatial_points_x, spatial_points_y, spatial_points_z], axes=(1, 2, 3))    
        return out
 

class Conv3dBlock(eqx.Module):
    conv: eqx.nn.Conv3d
    norm: eqx.nn.GroupNorm   
    dropout: eqx.nn.Dropout

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, p_do: float, *, key):
        padding = (kernel_size - 1) // 2
        self.conv = eqx.nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, use_bias=False, key=key)
        self.norm = eqx.nn.GroupNorm(groups=1, channels=out_channels, eps=1e-6, channelwise_affine=True)   
        self.dropout = eqx.nn.Dropout(p=p_do)

    def __call__(self, x, key=None, deterministic=False): 
        x = self.conv(x)
        x = self.norm(x)
        x = jax.nn.leaky_relu(x, negative_slope=0.1)
        if not deterministic:
            x = self.dropout(x, key=key)
        return x
    
class Deconv3dBlock(eqx.Module):
    deconv: eqx.nn.ConvTranspose3d

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, key):
        self.deconv = eqx.nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1, use_bias=True, key=key)

    def __call__(self, x: jnp.ndarray):
        x = self.deconv(x)
        x = jax.nn.leaky_relu(x, negative_slope=0.1)
        return x


class U_net(eqx.Module):
    conv1: Conv3dBlock
    conv2: Conv3dBlock
    conv2_add: Conv3dBlock
    conv3: Conv3dBlock
    conv3_add: Conv3dBlock
    deconv2: Deconv3dBlock
    deconv1: Deconv3dBlock
    deconv0: Deconv3dBlock
    output_layer: eqx.nn.Conv3d

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, dropout_rate: float, *,key):
        keys = jax.random.split(key, 9)
        self.conv1   = Conv3dBlock(input_channels,   output_channels, kernel_size, stride=2, p_do=dropout_rate, key=keys[0])
        self.conv2   = Conv3dBlock(input_channels,   output_channels, kernel_size, stride=2, p_do=dropout_rate, key=keys[1])
        self.conv2_add = Conv3dBlock(input_channels,   output_channels, kernel_size, stride=1, p_do=dropout_rate, key=keys[2])
        self.conv3   = Conv3dBlock(input_channels,   output_channels, kernel_size, stride=2, p_do=dropout_rate, key=keys[3])
        self.conv3_add = Conv3dBlock(input_channels,   output_channels, kernel_size, stride=1, p_do=dropout_rate, key=keys[4])

        self.deconv2 = Deconv3dBlock(output_channels, output_channels, kernel_size=kernel_size+1, key=keys[5])
        self.deconv1 = Deconv3dBlock(output_channels*2, output_channels, kernel_size=kernel_size+1, key=keys[6])
        self.deconv0 = Deconv3dBlock(output_channels*2, output_channels, kernel_size=kernel_size+1, key=keys[7])

        self.output_layer = eqx.nn.Conv3d(input_channels*2, output_channels, kernel_size=kernel_size, stride=1, padding=1, use_bias=True, key=keys[8])

    def __call__(self, x: jnp.ndarray, key=None, deterministic=False):
        if deterministic:
            keys = 5 * [None]
        else:   
            keys = jax.random.split(key, 5) 
        out_conv1 = self.conv1(x, key=keys[0], deterministic=deterministic)                                      
        out_conv2 = self.conv2_add(self.conv2(out_conv1, key=keys[2], deterministic=deterministic), key=keys[1], deterministic=deterministic)                   
        out_conv3 = self.conv3_add(self.conv3(out_conv2, key=keys[4], deterministic=deterministic), key=keys[3], deterministic=deterministic)                   
        out_deconv2 = self.deconv2(out_conv3)
        concat2   = jnp.concatenate([out_conv2, out_deconv2], axis=0)               
        out_deconv1   = self.deconv1(concat2)                                  
        concat1   = jnp.concatenate([out_conv1, out_deconv1], axis=0)
        out_deconv0   = self.deconv0(concat1)                                 
        concat0   = jnp.concatenate([x, out_deconv0], axis=0)
        return self.output_layer(concat0)
    

class UFNO3d(eqx.Module):
    in_channels: int
    out_channels: int
    width: int
    in_channels: int
    out_channels: int
    num_layers: int
    fc_lifting: eqx.nn.Linear
    fc_projection_0: eqx.nn.Linear
    fc_projection_1: eqx.nn.Linear
    spectral_convs: list[SpectralConv3d]
    bypass_convs: list[eqx.nn.Conv1d]
    unets: list[U_net]

    def __init__(self, in_channels:int, out_channels:int, num_layers: int, modes_x: int, modes_y: int, modes_z: int, width: int, p_do: float, *, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        keys = jax.random.split(key, 4)
        self.width = width
        self.fc_lifting   = eqx.nn.Linear(self.in_channels, width, key=keys[0]) 
        self.fc_projection_0 = eqx.nn.Linear(width, 1024, key=keys[1])   
        self.fc_projection_1 = eqx.nn.Linear(1024, self.out_channels, key=keys[2]) 

        self.spectral_convs = []
        self.bypass_convs = []
        self.unets =[]

        splitkey = keys[3]
        for i in range(self.num_layers):
            key1, key2, key3, splitkey = jax.random.split(splitkey, 4)  
            self.spectral_convs.append(SpectralConv3d(width, width, modes_x, modes_y, modes_z, key=key1)) 
            self.bypass_convs.append(eqx.nn.Conv1d(width, width, kernel_size=1, key=key2))
            self.unets.append(U_net(width, width, 3, p_do, key=key3))
    
    def __call__(self, x, key=None, deterministic=False):
        channels, spatial_points_x, spatial_points_y, spatial_points_z = x.shape
        x = jnp.transpose(x, (1, 2, 3, 0))
        x = jnp.pad(x, ((0,0), (0,8), (0,8), (0,0)), mode='edge')
        x = jnp.pad(x, ((0,8), (0,0), (0,0), (0,0)), mode='constant', constant_values=0)
        if not deterministic and key is None:
                raise ValueError("When running in nondeterministic mode key must be provided")
        if deterministic:
                    keys = self.num_layers * [None]
        else:
            keys = jax.random.split(key, self.num_layers)  

        x = jax.vmap(jax.vmap(jax.vmap(self.fc_lifting, in_axes=0), in_axes=0), in_axes=0)(x)
        x = jnp.transpose(x, (3, 0, 1, 2))

        for i in range(self.num_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.bypass_convs[i](x.reshape(self.width, -1)).reshape(self.width, spatial_points_x+8, spatial_points_y+8, spatial_points_z+8)
            x3 = self.unets[i](x, key=keys[i], deterministic=deterministic)
            x = x1 + x2 + x3
            x = jax.nn.relu(x)

        x = jnp.transpose(x, (1, 2, 3, 0)) 
        x = jax.vmap(jax.vmap(jax.vmap(self.fc_projection_0, in_axes=0), in_axes=0), in_axes=0)(x)
        x = jax.nn.relu(x)
        x = jax.vmap(jax.vmap(jax.vmap(self.fc_projection_1, in_axes=0), in_axes=0), in_axes=0)(x)        
        x = jnp.transpose(x, (3, 0, 1, 2))
        x = x.reshape(self.out_channels,spatial_points_x+8, spatial_points_y+8, spatial_points_z+8)[:,:-8,:-8,:-8]
        return x

