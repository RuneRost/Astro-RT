import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8" 
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=8"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import jax
import jax.numpy as jnp
from jax import random, vmap
import numpy as np
import equinox as eqx
import optax
import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as jshard
from jax.tree_util import tree_flatten
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib import colors

from architectures.ufno_3d import UFNO3d


def evaluate():
    
    # load your saved model (hyperparameters must match the saved model)
    modes = 4
    width = 16
    num_layers = 6
    batch_size = 8

    model = UFNO3d(in_channels=2, out_channels=1, num_layers=num_layers, modes_x=modes, modes_y=modes, modes_z=modes, width=width, p_do= 0.0, key=jax.random.PRNGKey(0))  
    model = eqx.tree_deserialise_leaves("surrogate_models/ufno_3d.eqx", model)

    print("Model loaded")

    # print out model info
    leaves, _ = tree_flatten(model)
    total_bytes = sum(x.size * x.dtype.itemsize for x in leaves if isinstance(x, jax.Array))
    print(f"Model size: {total_bytes / 1e6:.4f} MB")
    params = eqx.filter(model, eqx.is_array)
    total = sum(p.size for p in jax.tree_util.tree_leaves(params)) 
    print("Total params:", total)

    # create device mesh and sharding for multi gpu training 
    num_devices = len(jax.devices())
    devices = mesh_utils.create_device_mesh((num_devices, 1, 1, 1, 1))
    sharding = jshard.PositionalSharding(devices, )
    replicated = sharding.replicate()

    n_samples_test = inputs_test.shape[0]
    n_batches_test = jnp.ceil((n_samples_test/batch_size)).astype(int)

    # different loss functions
    def mse_loss(model, x, y, key=None, deterministic=False):
        y_pred = jax.vmap(model, in_axes=(0, None, None))(x, key, deterministic)  
        return jnp.mean(jnp.square(y_pred - y))
    
    def relative_loss(model, x, y, key=None, deterministic=False):
        y_pred = jax.vmap(model, in_axes=(0, None, None))(x, key, deterministic)  
        y_pred_flat = y_pred.reshape((-1))
        y_flat = y.reshape((-1))
        return jnp.mean(jnp.abs((y_flat - y_pred_flat)/y_flat))

    
    @eqx.filter_jit(donate="all-except-first")
    def evaluate_in_test(model, x, y, sharding):
        replicated = sharding.replicate()
        model = eqx.filter_shard(model, replicated)
        x, y = eqx.filter_shard((x, y), sharding)
        loss = relative_loss(model, x, y, deterministic=True)  
        return loss

    # compute loss on test set again
    test_loss = 0
    for i in range(0, inputs_test.shape[0], batch_size):
        if i + batch_size > inputs_test.shape[0]: # skip last batch to ensure sharding works 
            break
        batch_x = inputs_test[i:i + batch_size]
        batch_y = outputs_test[i:i + batch_size]
        batch_x, batch_y = eqx.filter_shard((batch_x, batch_y), sharding)
        test_loss += evaluate_in_test(model, batch_x, batch_y, sharding)
    test_loss /= (n_batches_test-1)
    print(f"Test Loss: {test_loss:.4f}")
    
    ## the following code can be used to recreate all plots shown in our paper  

    N = (outputs_test.shape[0]//batch_size)*batch_size
    M = outputs_test.shape[2] * outputs_test.shape[3] * outputs_test.shape[4]
    relative_flat = jnp.zeros(N * M)
    for i in range(0, inputs_test.shape[0], batch_size):
        if i + batch_size > inputs_test.shape[0]: # skip last batch so sharding works
            break
        batch_x = inputs_test[i:i + batch_size]
        batch_y = outputs_test[i:i + batch_size]
    
        batch_x, batch_y = eqx.filter_shard((batch_x, batch_y), sharding)
        replicated = sharding.replicate()
        model = eqx.filter_shard(model, replicated)
        key = None
        deterministic = True
        y_pred = jax.vmap(model, in_axes=(0, None, None))(batch_x, key, deterministic)
        relative = ((y_pred - batch_y)/(jnp.abs(batch_y))).reshape((-1))
        relative_flat  = relative_flat.at[i*M:(i+ batch_size)*M].add(relative)
    
    fig, ax = plt.subplots(1, 1, figsize=(30, 30))
    ax.ticklabel_format(style='plain', axis='y')
    ax.set_yticks([60000000, 120000000, 180000000, 240000000])
    ax.hist(relative_flat, bins=100, alpha=0.8, range=(-0.15, 0.15)) 
    ax.set_xlabel('Relative prediction error', fontsize=64)
    ax.set_ylabel("Number of pixels", fontsize=64)
    ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=48)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.tight_layout()
    plt.show()
    plt.savefig('histo_3d_relative_residual.pdf', bbox_inches='tight')

    fig, axes = plt.subplots(1, 3, figsize=(30, 30))

    im0 = axes[0].imshow(outputs_test[0, 0, :, :, 64//2], origin='lower', cmap='inferno',  vmin=0, vmax=1) 
    axes[0].set_xlabel("X", fontsize=30)
    axes[0].set_ylabel("Y", fontsize=30)
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].tick_params(axis='both', which='major',  length=20, width=3, labelsize=30)
    cb = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=24) 

    im1 = axes[1].imshow(model(inputs_test[0, :, :, :, :], deterministic=True)[0, :, :, 64//2], origin='lower', cmap='inferno',  vmin=0, vmax=1) 
    axes[1].set_xlabel("X", fontsize=30)
    axes[1].set_ylabel("Y", fontsize=30)
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[1].tick_params(axis='both', which='major',  length=20, width=3, labelsize=30)
    cb = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=24) 

    im2 = axes[2].imshow(model(inputs_test[0, :, :, :, :], deterministic=True)[0, :, :, 64//2] - outputs_test[0, 0, :, :, 64//2], origin='lower', cmap='coolwarm', vmin=-0.15, vmax=0.15) 
    axes[2].set_xlabel("X", fontsize=30)
    axes[2].set_ylabel("Y", fontsize=30)
    axes[2].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[2].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[2].tick_params(axis='both', which='major',  length=20, width=3, labelsize=30)
    cb = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=24) 
    cb.set_label(r"$\text{Prediction} - \text{Reference}$", fontsize=30)

    plt.tight_layout()
    plt.show()
    plt.savefig('3d_XY_plane.pdf', bbox_inches='tight')


    fig, axes = plt.subplots(1, 3, figsize=(30, 30))

    im0 = axes[0].imshow(outputs_test[0, 0, :, 64//2, :], origin='lower', cmap='inferno',  vmin=0, vmax=1) 
    axes[0].set_xlabel("X", fontsize=30)
    axes[0].set_ylabel("Y", fontsize=30)
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].tick_params(axis='both', which='major',  length=20, width=3, labelsize=30)
    cb = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=24) 

    im1 = axes[1].imshow(model(inputs_test[0, :, :, :, :], deterministic=True)[0, :, 64//2, :], origin='lower', cmap='inferno',  vmin=0, vmax=1) 
    axes[1].set_xlabel("X", fontsize=30)
    axes[1].set_ylabel("Y", fontsize=30)
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[1].tick_params(axis='both', which='major',  length=20, width=3, labelsize=30)
    cb = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=24) 

    im2 = axes[2].imshow(model(inputs_test[0, :, :, :, :], deterministic=True)[0, :, 64//2, :] - outputs_test[0, 0, :, 64//2, :], origin='lower', cmap='coolwarm', vmin=-0.15, vmax=0.15) 
    axes[2].set_xlabel("X", fontsize=30)
    axes[2].set_ylabel("Y", fontsize=30)
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[2].tick_params(axis='both', which='major',  length=20, width=3, labelsize=30)
    cb = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=24) 
    cb.set_label(r"$\text{Prediction} - \text{Reference}$", fontsize=30)
    plt.tight_layout()
    plt.show()
    plt.savefig('3d_XZ_plane.pdf', bbox_inches='tight')


    fig, axes = plt.subplots(1, 3, figsize=(30, 30))

    im0 = axes[0].imshow(outputs_test[0, 0, 64//2, :, :], origin='lower', cmap='inferno',  vmin=0, vmax=1) 
    axes[0].set_xlabel("X", fontsize=30)
    axes[0].set_ylabel("Y", fontsize=30)
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].tick_params(axis='both', which='major',  length=20, width=3, labelsize=30)
    cb = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=24) 


    im1 = axes[1].imshow(model(inputs_test[0, :, :, :, :], deterministic=True)[0, 64//2, :, :], origin='lower', cmap='inferno',  vmin=0, vmax=1) 
    axes[1].set_xlabel("X", fontsize=30)
    axes[1].set_ylabel("Y", fontsize=30)
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[1].tick_params(axis='both', which='major',  length=20, width=3, labelsize=30)
    cb = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=24) 

    im2 = axes[2].imshow(model(inputs_test[0, :, :, :, :], deterministic=True)[0, 64//2, :, :] - outputs_test[0, 0, 64//2, :, :], origin='lower', cmap='coolwarm', vmin=-0.15, vmax=0.15) 
    axes[2].set_xlabel("X", fontsize=30)
    axes[2].set_ylabel("Y", fontsize=30)
    axes[2].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[2].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[2].tick_params(axis='both', which='major',  length=20, width=3, labelsize=30)
    cb = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=24) 
    cb.set_label(r"$\text{Prediction} - \text{Reference}$", fontsize=30)

    plt.tight_layout()
    plt.show()
    plt.savefig('3d_YZ_plane.pdf', bbox_inches='tight')


    # plot to show aboprtion, emission and intensity of one sample
    inputs = inputs_test[4, :, :, :, :] * (xp_max[:, None, None, None] - xp_min[:, None, None, None]) + xp_min[:, None, None, None] -1e-8
    output = outputs_test[4, :, :, :, :] * (yp_max[:, None, None, None] - yp_min[:, None, None, None]) + yp_min[:, None, None, None] - 1e-8

    fig, ax = plt.subplots(figsize=(30, 30))
    im0 = ax.imshow(inputs[0, :, :, 64//2], origin='lower', cmap='viridis')  
    plt.tight_layout()
    plt.show()
    plt.savefig('a.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(30, 30))
    im1 = ax.imshow(inputs[1, :, :, 64//2], origin='lower', cmap='viridis')
    plt.tight_layout()
    plt.show()
    plt.savefig('j.pdf', bbox_inches='tight')


    fig, ax = plt.subplots(figsize=(30, 30))
    im2 = ax.imshow(output[0, :, :, 64//2], origin='lower', cmap='inferno')
    plt.tight_layout()
    plt.show()
    plt.savefig('I.pdf', bbox_inches='tight')


    fig, axes = plt.subplots(1, 3, figsize=(30, 30))
    unpreprocessed = inputs_test[0,:,:,:,:] * (xp_max[:, None, None, None] - xp_min[:, None, None, None]) + xp_min[:, None, None, None]
    im0 = axes[0].imshow(unpreprocessed[0, :, :, 64//2], origin='lower', cmap='viridis') #,  vmin=-2, vmax=1.5) 
    axes[0].set_xlabel("X", fontsize=36)
    axes[0].set_ylabel("Y", fontsize=36)

    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].tick_params(axis='both', which='major',  length=20, width=3, labelsize=24)
    cb = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb.set_label(r'$\log_{10}(a(x))$', fontsize=36)
    ticks = np.linspace(cb.vmin, cb.vmax, 5)  
    cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=30)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


    unpreprocessed = inputs_test[0,:,:,:,:] * (xp_max[:, None, None, None] - xp_min[:, None, None, None]) + xp_min[:, None, None, None]
    im1 = axes[1].imshow(unpreprocessed[1, :, :, 64//2], origin='lower', cmap='viridis') #,  vmin=2.5, vmax=5) 
    axes[1].set_xlabel("X", fontsize=36)
    axes[1].set_ylabel("Y", fontsize=36)
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[1].tick_params(axis='both', which='major',  length=20, width=3, labelsize=24)
    cb = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb.set_label(r'$\log_{10}(j(x))$', fontsize=36)
    ticks = np.linspace(cb.vmin, cb.vmax, 5)  
    cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=30)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    unpreprocessed  = outputs_test[0,:,:,:,:] * (yp_max[:, None, None, None] - yp_min[:, None, None, None]) + yp_min[:, None, None, None]
    im2 = axes[2].imshow(unpreprocessed[0, :, :, 64//2], origin='lower', cmap='inferno') #, vmin=-2.5, vmax=5.5) 
    axes[2].set_xlabel("X", fontsize=36)
    axes[2].set_ylabel("Y", fontsize=36)
    axes[2].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[2].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[2].tick_params(axis='both', which='major',  length=20, width=3, labelsize=24)
    cb = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cb.set_label(r'$\log_{10}(I(x))$', fontsize=36)
    ticks = np.linspace(cb.vmin, cb.vmax, 5)  
    cb.set_ticks(ticks)
    cb.ax.tick_params(length=20, width=3, labelsize=30)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.tight_layout()
    plt.show()
    plt.savefig('ajI_3d.pdf', bbox_inches='tight')

   
if __name__ == "__main__":
    print("Starting...")

    # in the README file there is a description on how to download the datsets

    inputs_test = np.load('datasets/inputs_test_3d.npy')
    outputs_test = np.load('datasets/outputs_test_3d.npy')

    # values with which data was preprocessed
    xp_min = jnp.array([-3.3366387, -8.000001])   
    xp_max = jnp.array([2.5393524, 4.189061])
    yp_min = jnp.array([-7.9991336])
    yp_max = jnp.array([5.1919284])

    evaluate()

    # additional code to measure prediction time of our model

    p_do = 0.07 
    modes = 4
    width = 16
    batch_size = 8
    model = UFNO3d(in_channels=2, out_channels=1, num_layers=6, modes_x=modes, modes_y=modes, modes_z=modes, width=width, p_do= p_do, key=jax.random.PRNGKey(0))  
    model = eqx.tree_deserialise_leaves("surrogate_models/ufno_3d.eqx", model)


    pred_fn = jax.jit(lambda x: model(x, deterministic=True))
    # warmup
    pred = pred_fn(inputs_test[0])

    # measure time
    t1 = time.time()
    pred = pred_fn(inputs_test[1])
    t2 = time.time()
    
    print(f"Prediction time for one sample: {t2-t1:.4f} seconds")

    # alternative approach (even faster)
    def step_fn(carry, _):
        input = carry
        output = model(input, deterministic=True)
        return input, output
    # warmup run
    final_state, inputs_all = jax.lax.scan(step_fn, inputs_test[0], xs=None, length=1)

    # measure time
    t3 = time.time()
    final_state, inputs_all = jax.lax.scan(step_fn, inputs_test[1], xs=None, length=1)
    t4 = time.time()

    print(f"Prediction time for one sample (faster approach): {t4-t3:.4f} seconds")



    

    


    
    
    
    
    

    