import jax
import jax.numpy as jnp
import equinox as eqx
from jax import vmap

class Conv3dBlock(eqx.Module):
    conv: eqx.nn.Conv3d
    norm: eqx.nn.GroupNorm   
    dropout: eqx.nn.Dropout

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, p_do: float, *, key):
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

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,*, key):
        self.deconv = eqx.nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride= stride, padding=padding, use_bias=True, key=key)


    def __call__(self, x: jnp.ndarray):
        x = self.deconv(x)
        x = jax.nn.leaky_relu(x, negative_slope=0.1)
        return x



class U_net(eqx.Module):
    conv1: Conv3dBlock
    conv1_1: Conv3dBlock
    conv1_2: Conv3dBlock
    conv2: Conv3dBlock
    conv2_1: Conv3dBlock
    conv2_2: Conv3dBlock
    conv3: Conv3dBlock
    conv3_1: Conv3dBlock
    conv3_2: Conv3dBlock
    conv4: Conv3dBlock
    conv4_1: Conv3dBlock
    conv4_2: Conv3dBlock
    deconv3: Deconv3dBlock
    deconv2: Deconv3dBlock
    deconv1: Deconv3dBlock
    deconv0: Deconv3dBlock
    output_layer1: eqx.nn.Conv3d
    output_layer2: eqx.nn.Conv3d

    def __init__(self, input_channels: int, output_channels: int, dropout_rate: float, *,key):
        keys = jax.random.split(key,17)
        self.conv1   = Conv3dBlock(input_channels,   output_channels, kernel_size=2, stride=2, padding=0, p_do=dropout_rate, key=keys[0])
        self.conv1_1 = Conv3dBlock(input_channels,   output_channels, kernel_size=2, stride=1, padding=0, p_do=dropout_rate, key=keys[1])
        self.conv1_2 = Conv3dBlock(input_channels,   output_channels, kernel_size=2, stride=1, padding=1, p_do=dropout_rate, key=keys[2])
        self.conv2   = Conv3dBlock(input_channels,   output_channels, kernel_size=2, stride=2, padding=0, p_do=dropout_rate, key=keys[3])
        self.conv2_1 = Conv3dBlock(input_channels,   output_channels, kernel_size=2, stride=1, padding=0, p_do=dropout_rate, key=keys[4])
        self.conv2_2 = Conv3dBlock(input_channels,   output_channels, kernel_size=2, stride=1, padding=1, p_do=dropout_rate, key=keys[5])
        self.conv3   = Conv3dBlock(input_channels,   output_channels, kernel_size=2, stride=2, padding=0, p_do=dropout_rate, key=keys[6])
        self.conv3_1 = Conv3dBlock(input_channels,   output_channels, kernel_size=2, stride=1, padding=0, p_do=dropout_rate, key=keys[7])
        self.conv3_2 = Conv3dBlock(input_channels,   output_channels, kernel_size=2, stride=1, padding=1, p_do=dropout_rate, key=keys[6])
        self.conv4   = Conv3dBlock(input_channels,   output_channels, kernel_size=2, stride=2, padding=0, p_do=dropout_rate, key=keys[8])
        self.conv4_1 = Conv3dBlock(input_channels,   output_channels, kernel_size=2, stride=1, padding=0, p_do=dropout_rate, key=keys[9])
        self.conv4_2 = Conv3dBlock(input_channels,   output_channels, kernel_size=2, stride=1, padding=1, p_do=dropout_rate, key=keys[10])
        self.deconv3 = Deconv3dBlock(output_channels, output_channels, kernel_size=3, stride=2, padding=0, key=keys[11])
        self.deconv2 = Deconv3dBlock(output_channels*2, output_channels, kernel_size=2, stride=2, padding=0, key=keys[12]) 
        self.deconv1 = Deconv3dBlock(output_channels*2, output_channels, kernel_size=2, stride=2, padding=0, key=keys[13])
        self.deconv0 = Deconv3dBlock(output_channels*2, output_channels, kernel_size=2, stride=2, padding=0, key=keys[14])

        self.output_layer1 = eqx.nn.Conv3d(input_channels*2, input_channels*2, kernel_size=2, stride=1, padding=1, use_bias=True, key=keys[15])
        self.output_layer2 = eqx.nn.Conv3d(input_channels*2, output_channels, kernel_size=2, stride=1, padding=0, use_bias=True, key=keys[16])

    def __call__(self, x: jnp.ndarray, key=None, deterministic=False):
        if deterministic:
            keys = 12 * [None]
        else:   
            keys = jax.random.split(key, 12) 
        out_conv1 = self.conv1_2(self.conv1_1(self.conv1(x, key=keys[0], deterministic=deterministic), key=keys[1], deterministic=deterministic),  key=keys[2], deterministic=deterministic)
        out_conv2 = self.conv2_2(self.conv2_1(self.conv2(out_conv1, key=keys[3], deterministic=deterministic), key=keys[4], deterministic=deterministic),  key=keys[5], deterministic=deterministic)
        out_conv3 = self.conv3_2(self.conv3_1(self.conv3(out_conv2, key=keys[6], deterministic=deterministic), key=keys[7], deterministic=deterministic), key=keys[8], deterministic=deterministic)        
        out_conv4 = self.conv4_2(self.conv4_1(self.conv4(out_conv3, key=keys[9], deterministic=deterministic), key=keys[10], deterministic=deterministic), key=keys[11], deterministic=deterministic)  
        out_deconv3 = self.deconv3(out_conv4)
        concat3   = jnp.concatenate([out_conv3, out_deconv3], axis=0)              
        out_deconv2   = self.deconv2(concat3)
        concat2   = jnp.concatenate([out_conv2, out_deconv2], axis=0)              
        out_deconv1   = self.deconv1(concat2) 
        concat1   = jnp.concatenate([out_conv1, out_deconv1], axis=0)
        out_deconv0   = self.deconv0(concat1)  
        concat0   = jnp.concatenate([x, out_deconv0], axis=0)
        return self.output_layer2(self.output_layer1(concat0))
    
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
    

    def __init__(
            self,
            in_channels,
            out_channels,
            modes_x,
            modes_y,
            modes_z,
            *,
            key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z

        scale = jnp.sqrt(1.0 / (in_channels + out_channels)) #some other side used this

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
    
class SimpleBlock3d(eqx.Module):
    width: int
    in_channels: int
    out_channels: int
    fc0: eqx.nn.Linear
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear
    conv0: SpectralConv3d
    conv1: SpectralConv3d
    conv2: SpectralConv3d
    conv3: SpectralConv3d
    conv4: SpectralConv3d
    conv5: SpectralConv3d
    w0: eqx.nn.Conv1d
    w1: eqx.nn.Conv1d
    w2: eqx.nn.Conv1d
    w3: eqx.nn.Conv1d
    w4: eqx.nn.Conv1d
    w5: eqx.nn.Conv1d
    unet0: U_net
    unet1: U_net
    unet2: U_net
    unet3: U_net
    unet4: U_net
    unet5: U_net

    def __init__(self, in_channels:int, out_channels:int, modes_x: int, modes_y: int, modes_z: int, width: int, p_do: float, *, key):
        keys = jax.random.split(key, 22)
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc0   = eqx.nn.Linear(self.in_channels, width, key=keys[0])  
        self.conv0 = SpectralConv3d(width, width, modes_x, modes_y, modes_z, key=keys[1])
        self.conv1 = SpectralConv3d(width, width, modes_x, modes_y, modes_z, key=keys[2])
        self.conv2 = SpectralConv3d(width, width, modes_x, modes_y, modes_z, key=keys[3])
        self.conv3 = SpectralConv3d(width, width, modes_x, modes_y, modes_z, key=keys[4])
        self.conv4 = SpectralConv3d(width, width, modes_x, modes_y, modes_z, key=keys[5])
        self.conv5 = SpectralConv3d(width, width, modes_x, modes_y, modes_z, key=keys[6])
        self.w0 = eqx.nn.Conv1d(width, width, kernel_size=1, key=keys[7])
        self.w1 = eqx.nn.Conv1d(width, width, kernel_size=1, key=keys[8])
        self.w2 = eqx.nn.Conv1d(width, width, kernel_size=1, key=keys[9])
        self.w3 = eqx.nn.Conv1d(width, width, kernel_size=1, key=keys[10])
        self.w4 = eqx.nn.Conv1d(width, width, kernel_size=1, key=keys[11])
        self.w5 = eqx.nn.Conv1d(width, width, kernel_size=1, key=keys[12])
        self.unet0 = U_net(width, width, p_do, key=keys[13]) 
        self.unet1 = U_net(width, width, p_do, key=keys[14]) 
        self.unet2 = U_net(width, width, p_do, key=keys[15]) 
        self.unet3 = U_net(width, width, p_do, key=keys[16]) 
        self.unet4 = U_net(width, width, p_do, key=keys[17])
        self.unet5 = U_net(width, width, p_do, key=keys[18]) 
        self.fc1 = eqx.nn.Linear(width, 128, key=keys[19])   
        self.fc2 = eqx.nn.Linear(128, 512, key=keys[20])  
        self.fc3 = eqx.nn.Linear(512, self.out_channels, key=keys[21])  

    def __call__(self, x: jnp.ndarray, key=None, deterministic=False):
        if deterministic:
            keys = 6 * [None]
        else:
            keys = jax.random.split(key, 6)
        spatial_points_x, spatial_points_y, spatial_points_z, channels = x.shape   
        
        
        x = jax.vmap(jax.vmap(jax.vmap(self.fc0, in_axes=0), in_axes=0), in_axes=0)(x)

        x = jnp.transpose(x, (3, 0, 1, 2))

        x1 = self.conv0(x)
        x2 = self.w0(x.reshape(self.width, -1)).reshape(self.width, spatial_points_x, spatial_points_y, spatial_points_z)
        x3 = self.unet0(x, key=keys[0], deterministic=deterministic)
        x = x1 + x2 + x3
        x = jax.nn.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.reshape(self.width, -1)).reshape(self.width, spatial_points_x, spatial_points_y, spatial_points_z)
        x3 = self.unet1(x, key=keys[1], deterministic=deterministic)
        x = x1 + x2 + x3
        x = jax.nn.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.reshape(self.width, -1)).reshape(self.width, spatial_points_x, spatial_points_y, spatial_points_z)
        x3 = self.unet2(x, key=keys[2], deterministic=deterministic)
        x = x1 + x2 + x3
        x = jax.nn.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.reshape(self.width, -1)).reshape(self.width, spatial_points_x, spatial_points_y, spatial_points_z)
        x3 = self.unet3(x, key=keys[3], deterministic=deterministic) 
        x = x1 + x2 + x3
        x = jax.nn.relu(x)

        x1 = self.conv4(x)
        x2 = self.w4(x.reshape(self.width, -1)).reshape(self.width, spatial_points_x, spatial_points_y, spatial_points_z)
        x3 = self.unet4(x, key=keys[4], deterministic=deterministic)
        x = x1 + x2 + x3
        x = jax.nn.relu(x)
        
        x1 = self.conv5(x)
        x2 = self.w5(x.reshape(self.width, -1)).reshape(self.width, spatial_points_x, spatial_points_y, spatial_points_z)
        x3 = self.unet5(x, key=keys[5], deterministic=deterministic)
        x = x1 + x2 + x3
        x = jax.nn.relu(x)
        
        x = jnp.transpose(x, (1, 2, 3, 0)) 

        x = jax.vmap(jax.vmap(jax.vmap(self.fc1, in_axes=0), in_axes=0), in_axes=0)(x)
        x = jax.nn.relu(x)
        x = jax.vmap(jax.vmap(jax.vmap(self.fc2, in_axes=0), in_axes=0), in_axes=0)(x) 
        x = jax.nn.relu(x)
        x = jax.vmap(jax.vmap(jax.vmap(self.fc3, in_axes=0), in_axes=0), in_axes=0)(x)        
        x = jnp.transpose(x, (3, 0, 1, 2))            
        return x
    

class UFNO3d(eqx.Module):
    conv1: SimpleBlock3d
    in_channels: int
    out_channels: int

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int, modes_z: int, width: int, p_do: float, *, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = SimpleBlock3d(in_channels, out_channels, modes_x, modes_y, modes_z, width, p_do, key=key)
    

    def __call__(self, x, key=None, deterministic=False):
        channels, spatial_points_x, spatial_points_y, spatial_points_z = x.shape
        x = jnp.transpose(x, (1, 2, 3, 0))
        x = jnp.pad(x, ((0,0), (0,8), (0,8), (0,0)), mode='edge')
        x = jnp.pad(x, ((0,8), (0,0), (0,0), (0,0)), mode='constant', constant_values=0)
        if not deterministic and key is None:
                raise ValueError("When running in nondeterministic mode key must be provided")
        x = self.conv1(x, key=key, deterministic=deterministic)
        x = x.reshape(self.out_channels,spatial_points_x+8, spatial_points_y+8, spatial_points_z+8)[:,:-8,:-8,:-8]
        return x

    def count_params(self):
        leaves = jax.tree_util.tree_leaves(self)
        total = 0
        for leaf in leaves:
            if isinstance(leaf, jnp.ndarray):
                total += leaf.size
        return total
