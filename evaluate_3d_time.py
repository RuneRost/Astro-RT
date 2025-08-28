import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12" 
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=12"
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
import matplotlib
import matplotlib.pyplot as plt
import imageio
import time
import sys
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter

from architectures.ufno_3d_time import UFNO3d


def evaluate(inputs_test, outputs_test):
    
    # load your saved model (hyperparameters must match the saved model)
    modes           = 4 
    width           = 32
    num_layers      = 6
    batch_size = 8

    # load predefine model 
    model = UFNO3d(in_channels=3, out_channels=1, num_layers=num_layers, modes_x=modes, modes_y=modes, modes_z=modes, width= width, p_do=0.0, key=jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves("surrogate_models/ufno_3d_time.eqx", model)

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
        rel_err = jnp.abs((y_flat - y_pred_flat)/(y_flat + 1e-12))
        mask = (y_flat != 0)  # ignoring points where y is exactly zero to avoid division by zero
        rel_err_masked = rel_err * mask   
        mean_error = jnp.sum(rel_err_masked) / jnp.sum(mask)  # mean relative error over non-zero points
        return mean_error
        
    @eqx.filter_jit(donate="all-except-first")
    def evaluate_in_test(model, x, y, sharding):
        replicated = sharding.replicate()
        model = eqx.filter_shard(model, replicated)
        x, y = eqx.filter_shard((x, y), sharding)
        loss = relative_loss(model, x, y, deterministic=True)  
        return loss
    
    model = eqx.filter_shard(model, replicated)  
 
    test_loss = 0
    for i in range(0, inputs_test.shape[0], batch_size):
        if i + batch_size > inputs_test.shape[0]: # skip last batch to ensure sharding works 
            break
        batch_x = inputs_test[i:i + batch_size]
        batch_y = outputs_test[i:i + batch_size]
        batch_x, batch_y = eqx.filter_shard((batch_x, batch_y), sharding)
        test_loss += evaluate_in_test(model, batch_x, batch_y, sharding)
    test_loss /= (n_batches_test-1)
    print(f"Test Loss (Relative Loss): {test_loss:.4f}")

    ## the following code can be used to recreate all plots shown in our paper    

    # to create gifs, png format is needed, if you want pdf format, please change the file extension from .png to .pdf and remove the gif related code
    filenames_true = []
    fig, ax = plt.subplots(figsize=(30, 30)) 
    im = ax.imshow(inputs_test[18, 2, :, :, 64//2], origin='lower', cmap='inferno',  vmin=0, vmax=1)
    ax.set_xlabel("x", fontsize=130)
    ax.set_ylabel("y", fontsize=130)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=90)
    plt.tight_layout()
    filename = 'true_XY_init.png'
    plt.savefig(filename, bbox_inches='tight')
    filenames_true.append(filename)
    plt.close()

    for step in range(9):  
        fig, ax = plt.subplots(figsize=(30, 30)) 
        im = ax.imshow(outputs_test[18 + step, 0, :, :, 64//2], origin='lower', cmap='inferno',  vmin=0, vmax=1)
        ax.set_xlabel("x", fontsize=130)
        ax.set_ylabel("y", fontsize=130)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=90)
        plt.tight_layout()
        filename = f'true_XY_{step:03d}.png'
        plt.savefig(filename, bbox_inches='tight')
        filenames_true.append(filename)
        plt.close()

    gif_filename = 'true_XY.gif'
    with imageio.get_writer(gif_filename, mode='I', duration=10.0, loop = 0) as writer:
        for filename in filenames_true:
            image = imageio.imread(filename)
            writer.append_data(image)

    filenames_pred = []
    fig, ax = plt.subplots(figsize=(30, 30))
    im = ax.imshow(inputs_test[18, 2, :, :, 64//2], origin='lower', cmap='inferno',  vmin=0, vmax=1) # timestep 1
    ax.set_xlabel("x", fontsize=130)
    ax.set_ylabel("y", fontsize=130)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=90)
    plt.tight_layout()
    filename = 'pred_XY_init.png'
    plt.savefig(filename, bbox_inches='tight')
    filenames_pred.append(filename)
    plt.close()

    input = jnp.array(inputs_test[18, :, :, :, :]) 
    for step in range(9):
        output = model(input, deterministic=True)
        np.save(f"output{step:03d}.npy", output[0,:,:,:])  
        input = input.at[2,:,:,:].set(output[0,:,:,:])
        fig, ax = plt.subplots(figsize=(30, 30))
        im = ax.imshow(output[0,:,:,64//2], origin='lower', cmap='inferno',  vmin=0, vmax=1)
        ax.set_xlabel("x", fontsize=130)
        ax.set_ylabel("y", fontsize=130)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=90)
        plt.tight_layout()
        filename = f'pred_XY_{step:03d}.png'
        plt.savefig(filename, bbox_inches='tight')
        filenames_pred.append(filename)
        plt.close()
    
    gif_filename = 'pred_XY.gif'
    with imageio.get_writer(gif_filename, mode='I', duration=10.0, loop = 0) as writer:
        for filename in filenames_pred:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    filenames_res = []
    fig, ax = plt.subplots(figsize=(30, 30))
    im = ax.imshow(inputs_test[18, 2, :, :, 64//2] - inputs_test[18, 2, :, :, 64//2], origin='lower', cmap='coolwarm',  vmin=-0.2, vmax=0.2) # timestep 1
    ax.set_xlabel("x", fontsize=130)
    ax.set_ylabel("y", fontsize=130)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=90)
    plt.tight_layout()
    filename = f'res_XY_init.png'
    plt.savefig(filename, bbox_inches='tight')
    filenames_res.append(filename)
    plt.close()

    input = jnp.array(inputs_test[18, :, :, :, :]) 
    for step in range(9): 
        output = model(input, deterministic=True)
        input = input.at[2,:,:,:].set(output[0,:,:,:])
        fig, ax = plt.subplots(figsize=(30, 30))
        im = ax.imshow(output[0,:,:,64//2] - outputs_test[18 + step, 0, :, :, 64//2], origin='lower', cmap='coolwarm',  vmin=-0.2, vmax=0.2)
        ax.set_xlabel("x", fontsize=130)
        ax.set_ylabel("y", fontsize=130)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=90)
        plt.tight_layout()
        filename = f'res_XY_{step:03d}.png'
        plt.savefig(filename, bbox_inches='tight')
        filenames_res.append(filename)
        plt.close()
    
    gif_filename = 'res_XY.gif'
    with imageio.get_writer(gif_filename, mode='I', duration=10.0, loop = 0) as writer:
        for filename in filenames_res:
            image = imageio.imread(filename)
            writer.append_data(image)
      
    filenames_true = []
    fig, ax = plt.subplots(figsize=(30, 30)) 
    im = ax.imshow(inputs_test[18, 2, :, 64//2, :], origin='lower', cmap='inferno',  vmin=0, vmax=1)
    ax.set_xlabel("X", fontsize=130)
    ax.set_ylabel("Y", fontsize=130)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=90)
    plt.tight_layout()
    filename = 'true_XZ_init.png'
    plt.savefig(filename, bbox_inches='tight')
    filenames_true.append(filename)
    plt.close()

    for step in range(9): 
        fig, ax = plt.subplots(figsize=(30, 30)) 
        im = ax.imshow(outputs_test[18 + step, 0, :, 64//2, :], origin='lower', cmap='inferno',  vmin=0, vmax=1)
        ax.set_xlabel("x", fontsize=130)
        ax.set_ylabel("y", fontsize=130)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=90)
        plt.tight_layout()
        filename = f'true_XZ_{step:03d}.png'
        plt.savefig(filename, bbox_inches='tight')
        filenames_true.append(filename)
        plt.close()

    gif_filename = 'true_XZ.gif'
    with imageio.get_writer(gif_filename, mode='I', duration=10.0, loop = 0) as writer:
        for filename in filenames_true:
            image = imageio.imread(filename)
            writer.append_data(image)

    filenames_pred = []
    fig, ax = plt.subplots(figsize=(30, 30))
    im = ax.imshow(inputs_test[18, 2, :, 64//2, :], origin='lower', cmap='inferno',  vmin=0, vmax=1) # timestep 1
    ax.set_xlabel("x", fontsize=130)
    ax.set_ylabel("y", fontsize=130)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=90)
    plt.tight_layout()
    filename = 'pred_XZ_init.png'
    plt.savefig(filename, bbox_inches='tight')
    filenames_pred.append(filename)
    plt.close()

    input = jnp.array(inputs_test[18, :, :, :, :]) 
    for step in range(9): 
        output = model(input, deterministic=True)
        input = input.at[2,:,:,:].set(output[0,:,:,:])
        fig, ax = plt.subplots(figsize=(30, 30))
        im = ax.imshow(output[0,:,64//2,:], origin='lower', cmap='inferno',  vmin=0, vmax=1)
        ax.set_xlabel("x", fontsize=130)
        ax.set_ylabel("y", fontsize=130)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=90)
        plt.tight_layout()
        filename = f'pred_XZ_{step:03d}.png'
        plt.savefig(filename, bbox_inches='tight')
        filenames_pred.append(filename)
        plt.close()

    gif_filename = 'pred_XZ.gif'
    with imageio.get_writer(gif_filename, mode='I', duration=10.0, loop = 0) as writer:
        for filename in filenames_pred:
            image = imageio.imread(filename)
            writer.append_data(image)

    filenames_res = []
    fig, ax = plt.subplots(figsize=(30, 30))
    im = ax.imshow(inputs_test[18, 2, :, 64//2, :] - inputs_test[18, 2, :, 64//2, :], origin='lower', cmap='coolwarm',  vmin=-0.2, vmax=0.2) # timestep 1
    ax.set_xlabel("x", fontsize=130)
    ax.set_ylabel("y", fontsize=130)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=90)
    plt.tight_layout()
    filename = f'res_XZ_init.png'
    plt.savefig(filename, bbox_inches='tight')
    filenames_pred.append(filename)
    plt.close()

    input = jnp.array(inputs_test[18, :, :, :, :]) 
    for step in range(9): 
        output = model(input, deterministic=True)
        input = input.at[2,:,:,:].set(output[0,:,:,:])
        fig, ax = plt.subplots(figsize=(30, 30))
        im = ax.imshow(output[0,:,64//2,:] - outputs_test[18 + step, 0, :, 64//2, :], origin='lower', cmap='coolwarm',  vmin=-0.2, vmax=0.2)
        ax.set_xlabel("x", fontsize=130)
        ax.set_ylabel("y", fontsize=130)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=90)
        plt.tight_layout()
        filename = f'res_XZ_{step:03d}.png'
        plt.savefig(filename, bbox_inches='tight')
        filenames_pred.append(filename)
        plt.close()

    gif_filename = 'res_XZ.gif'
    with imageio.get_writer(gif_filename, mode='I', duration=10.0, loop = 0) as writer:
        for filename in filenames_res:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    filenames_true = []
    fig, ax = plt.subplots(figsize=(30, 30)) 
    im = ax.imshow(inputs_test[18, 2, 64//2, :, :], origin='lower', cmap='inferno',  vmin=0, vmax=1)
    ax.set_xlabel("x", fontsize=130)
    ax.set_ylabel("y", fontsize=130)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=90)
    plt.tight_layout()
    filename = 'true_YZ_init.png'
    plt.savefig(filename, bbox_inches='tight')
    filenames_true.append(filename)
    plt.close()

    for step in range(9):  
        fig, ax = plt.subplots(figsize=(30, 30)) 
        im = ax.imshow(outputs_test[18 + step, 0, 64//2, :, :], origin='lower', cmap='inferno',  vmin=0, vmax=1)
        ax.set_xlabel("x", fontsize=130)
        ax.set_ylabel("y", fontsize=130)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=90)
        plt.tight_layout()
        filename = f'true_YZ_{step:03d}.png'
        plt.savefig(filename, bbox_inches='tight')
        filenames_true.append(filename)
        plt.close()

    gif_filename = 'true_YZ.gif'
    with imageio.get_writer(gif_filename, mode='I', duration=10.0, loop = 0) as writer:
        for filename in filenames_true:
            image = imageio.imread(filename)
            writer.append_data(image)

    filenames_pred = []
    fig, ax = plt.subplots(figsize=(30, 30))
    im = ax.imshow(inputs_test[18, 2, 64//2, :, :], origin='lower', cmap='inferno',  vmin=0, vmax=1) # timestep 1
    ax.set_xlabel("x", fontsize=130)
    ax.set_ylabel("y", fontsize=130)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=90)
    plt.tight_layout()
    filename = 'pred_YZ_init.png'
    plt.savefig(filename, bbox_inches='tight')
    filenames_pred.append(filename)
    plt.close()

    input = jnp.array(inputs_test[18, :, :, :, :]) 
    for step in range(9): 
        output = model(input, deterministic=True)
        input = input.at[2,:,:,:].set(output[0,:,:,:])
        fig, ax = plt.subplots(figsize=(30, 30))
        im = ax.imshow(output[0,64//2,:,:], origin='lower', cmap='inferno',  vmin=0, vmax=1)
        ax.set_xlabel("x", fontsize=130)
        ax.set_ylabel("y", fontsize=130)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=90)
        plt.tight_layout()
        filename = f'pred_YZ_{step:03d}.png'
        plt.savefig(filename, bbox_inches='tight')
        filenames_pred.append(filename)
        plt.close()

    gif_filename = 'pred_YZ.gif'
    with imageio.get_writer(gif_filename, mode='I', duration=10.0, loop = 0) as writer:
        for filename in filenames_pred:
            image = imageio.imread(filename)
            writer.append_data(image)

    filenames_res = []
    fig, ax = plt.subplots(figsize=(30, 30))
    im = ax.imshow(inputs_test[18, 2, 64//2, :, :] - inputs_test[18, 2, 64//2, :, :], origin='lower', cmap='coolwarm',  vmin=-0.2, vmax=0.2) # timestep 1
    ax.set_xlabel("x", fontsize=130)
    ax.set_ylabel("y", fontsize=130)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=90)
    plt.tight_layout()
    filename = f'res_YZ_init.png'
    plt.savefig(filename, bbox_inches='tight')
    filenames_pred.append(filename)
    plt.close()

    input = jnp.array(inputs_test[18, :, :, :, :]) 
    for step in range(9): 
        output = model(input, deterministic=True)
        input = input.at[2,:,:,:].set(output[0,:,:,:])
        fig, ax = plt.subplots(figsize=(30, 30))
        im = ax.imshow(output[0,64//2,:,:] - outputs_test[18 + step, 0, 64//2, :, :], origin='lower', cmap='coolwarm',  vmin=-0.2, vmax=0.2)
        ax.set_xlabel("x", fontsize=130)
        ax.set_ylabel("y", fontsize=130)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=90)
        plt.tight_layout()
        filename = f'res_YZ_{step:03d}.png'
        plt.savefig(filename, bbox_inches='tight')
        filenames_pred.append(filename)
        plt.close()

    gif_filename = 'res_YZ.gif'
    with imageio.get_writer(gif_filename, mode='I', duration=10.0, loop = 0) as writer:
        for filename in filenames_res:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    cmap = plt.get_cmap("inferno")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  
    fig = plt.figure(figsize=(2, 8)) 
    ax = fig.add_axes([0.15, 0.05, 0.2, 0.9]) 
    cbar = fig.colorbar(sm, cax=ax)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=24)
    plt.savefig("colorbar_num.png", bbox_inches=None, pad_inches=0.0)
    plt.close(fig)

    cmap = plt.get_cmap("inferno")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  
    fig = plt.figure(figsize=(2, 8)) 
    ax = fig.add_axes([0.15, 0.05, 0.2, 0.9])
    cbar = fig.colorbar(sm, cax=ax)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=24)
    plt.savefig("colorbar_pred.png", bbox_inches=None, pad_inches=0.0)
    plt.close(fig)

    cmap = plt.get_cmap("coolwarm")
    norm = matplotlib.colors.Normalize(vmin=-0.2, vmax=0.2)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  
    fig = plt.figure(figsize=(2, 8)) 
    ax = fig.add_axes([0.15, 0.05, 0.2, 0.9])
    cbar = fig.colorbar(sm, cax=ax)
    cbar.set_ticks([-0.2, 0.0, 0.2])
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label(r"$\text{Prediction} - \text{Reference}$", fontsize=30)
    plt.savefig("colorbar_res.pdf") 
    plt.close(fig)
    
    fig, axes = plt.subplots(1, 4, figsize=(30, 30))  
    inputs = inputs_test[18, :, :, :, :]  * (xp_max[:, None, None, None] - xp_min[:, None, None, None]) + xp_min[:, None, None, None]
    output_5 = outputs_test[18+5, :, :, :, :] * (yp_max[:, None, None, None] - yp_min[:, None, None, None]) + yp_min[:, None, None, None]
    output_6 = outputs_test[18+6, :, :, :, :] * (yp_max[:, None, None, None] - yp_min[:, None, None, None]) + yp_min[:, None, None, None]

    im0 = axes[0].imshow(inputs[0, :, :, 64//2], origin='lower', cmap='viridis') 
    axes[0].set_xlabel("x", fontsize=36)
    axes[0].set_ylabel("y", fontsize=36)
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].tick_params(axis='both', which='major',  length=20, width=3, labelsize=24)
    cb = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb.set_label(r'$\log_{10}(a(x))$', fontsize=36)
    ticks = np.linspace(cb.vmin, cb.vmax, 5)  
    cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=30)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    im1 = axes[1].imshow(inputs[1, :, :, 64//2], origin='lower', cmap='viridis') 
    axes[1].set_xlabel("x", fontsize=36)
    axes[1].set_ylabel("y", fontsize=36)
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[1].tick_params(axis='both', which='major',  length=20, width=3, labelsize=24)
    cb = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb.set_label(r'$\log_{10}(j(x))$', fontsize=36)
    ticks = np.linspace(cb.vmin, cb.vmax, 5)  
    cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=30)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    im2 = axes[2].imshow(output_5[0, :, :, 64//2], origin='lower', cmap='inferno')
    axes[2].set_xlabel("x", fontsize=36)
    axes[2].set_ylabel("y", fontsize=36)
    axes[2].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[2].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[2].tick_params(axis='both', which='major',  length=20, width=3, labelsize=24)
    cb = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cb.set_label(r'$\log_{10}(I_5(x))$', fontsize=36)
    ticks = np.linspace(cb.vmin, cb.vmax, 5)  
    cb.set_ticks(ticks)
    cb.ax.tick_params(length=20, width=3, labelsize=30)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    im3 = axes[3].imshow(output_6[0, :, :, 64//2], origin='lower', cmap='inferno') #,  vmin=-8, vmax=7) 
    axes[3].set_xlabel("x", fontsize=36)
    axes[3].set_ylabel("y", fontsize=36)
    axes[3].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[3].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[3].tick_params(axis='both', which='major',  length=20, width=3, labelsize=24)
    cb = fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    cb.set_label(r'$\log_{10}(I_6(x))$', fontsize=36)
    ticks = np.linspace(cb.vmin, cb.vmax, 5)  
    cb.set_ticks(ticks)
    cb.ax.tick_params(length=20, width=3, labelsize=30)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.tight_layout()
    plt.show()
    plt.savefig('ajI_3d_time.pdf', bbox_inches='tight')

    inputs_test= inputs_test[::9]
    outputs_test = outputs_test.reshape(-1, 9, outputs_test.shape[2], outputs_test.shape[3], outputs_test.shape[4])

    N = (outputs_test.shape[0]//batch_size)*batch_size
    M = outputs_test.shape[2] * outputs_test.shape[3] * outputs_test.shape[4]
    residuals_flat = jnp.zeros((10, N * M))
    residuals_flat = residuals_flat.at[0, :].set(jnp.inf *jnp.ones(N * M))  
    for i in range(0, inputs_test.shape[0], batch_size):
        if i + batch_size > inputs_test.shape[0]: # skip last batch so sharding works
                break
        batch_x = inputs_test[i:i+ batch_size]
        batch_y = outputs_test[i:i+batch_size]
        batch_x, batch_y = eqx.filter_shard((batch_x, batch_y), sharding)
        replicated = sharding.replicate()
        model = eqx.filter_shard(model, replicated)
        key = None
        deterministic = True
        temporal_I = jax.vmap(model, in_axes=(0, None, None))(batch_x, key, deterministic).reshape((-1, outputs_test.shape[2], outputs_test.shape[3], outputs_test.shape[4]))
        batch_x = batch_x.at[:, 2, :, :, :].set(temporal_I)
        residuals = ((temporal_I.reshape((-1)) - batch_y[:, 0, :, :, :].reshape((-1)))/jnp.abs(batch_y[:, 0, :, :, :].reshape((-1))))
        mask = (batch_y[:, 0, :, :, :].reshape((-1)) != 0)
        residuals= jnp.where(mask, residuals, jnp.inf)
        residuals_flat  = residuals_flat.at[1, i*M:(i+ batch_size)*M].set(residuals)
        for j in range(8):
            temporal_I = jax.vmap(model, in_axes=(0, None, None))(batch_x, key, deterministic).reshape((-1, outputs_test.shape[2], outputs_test.shape[3], outputs_test.shape[4]))
            batch_x = batch_x.at[:, 2, :, :, :].set(temporal_I)
            residuals = ((temporal_I.reshape((-1)) - batch_y[:,j+1, :, :, :].reshape((-1)))/jnp.abs(batch_y[:, j+1, :, :, :].reshape((-1))))
            mask = (batch_y[:, j+1, :, :, :].reshape((-1)) != 0)
            residuals = jnp.where(mask, residuals, jnp.inf)
            residuals_flat  = residuals_flat.at[j+2, i*M:(i+ batch_size)*M].set(residuals)
    
    for i in range(10):
        fig, ax = plt.subplots(1, 1, figsize=(30, 30))
        ax.ticklabel_format(style='plain', axis='y') 
        ax.set_ylim(0, 540000)   
        ax.set_yticks([100000, 200000, 300000, 400000, 500000])
        ax.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2])
        ax.hist(residuals_flat[i], bins=100, alpha=0.8, range=(-0.2, 0.2)) 
        ax.set_xlabel('Relative prediction error', fontsize=90)
        ax.set_ylabel("Number of pixels", fontsize=90)
        ax.tick_params(axis='both', which='major',  length=20, width=3, labelsize=60)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        plt.tight_layout()
        plt.show()
        plt.savefig(f'histo_3d_time_relative_residual_{i}.pdf', bbox_inches='tight')

    return test_loss
    




   
if __name__ == "__main__":
    print("Starting...")

    # in the README file there is a description on how to download the datsets

    inputs_test = np.load('datasets/inputs_test_3d_time.npy')
    outputs_test = np.load('datasets/outputs_test_3d_time.npy')

    print("Data has been loaded")

    # values with which data was preprocessed
    xp_min = jnp.array([-1.7541726, -8.000001, -8.000001]) 
    xp_max = jnp.array([0.81056046, 4.197271, 5.4913197]) 
    yp_min = jnp.array([-8.000001]) 
    yp_max = jnp.array([5.4913197])

    evaluate(inputs_test, outputs_test)

    # additional code to measure prediction time of our model
    
    
    modes           = 4 
    width           = 32
    num_layers      = 6

    # load predefine model 
    model = UFNO3d(in_channels=3, out_channels=1, num_layers=num_layers, modes_x=modes, modes_y=modes, modes_z=modes, width= width, p_do=0.0, key=jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves("surrogate_models/ufno_3d_time.eqx", model)

    def step_fn(carry, _):
        input = carry
        output = model(input, deterministic=True)
        input = input.at[2, :, :, :].set(output[0, :, :, :])
        return input, input   

    input0 = jnp.array(inputs_test[0, :, :, :, :])

    #warmup
    final_state, inputs_all = jax.lax.scan(step_fn, inputs_test[0, :, :, :, :], xs=None, length=9)

    # measure time
    t1 = time.time()
    final_state, inputs_all = jax.lax.scan(step_fn, inputs_test[9, :, :, :, :], xs=None, length=9) # every 9th sample in inputs_test represents a starting position
    t2 = time.time()

    print(f"Prediction time for full temporal evolution of one sample: {t2-t1:.4f} seconds")  