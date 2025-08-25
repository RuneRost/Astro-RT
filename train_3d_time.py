import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12" 
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=12"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import equinox as eqx
import optax
import optuna
import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as jshard
from jax.tree_util import tree_flatten
import imageio
import time

from architectures.ufno_3d_time import UFNO3d


def objective(trial):
    
    # define regions in which hyperparameters should be optimized -> currently commented out to use fixed values found by optimal parameter search
    #lr_start        = trial.suggest_float("lr_start", 3e-4, 7e-4, log=True)
    #dr              = trial.suggest_float("decay_rate", 0.87, 0.93)
    #wd              = trial.suggest_float("wd", 3e-4, 7e-4)
    #p_do            = trial.suggest_categorical("p_do", [0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.10])
    #p_do            = trial.suggest_categorical("p_do", [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.10, 0.105, 0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15])
    #modes           = 4 #trial.suggest_int("modes", 4, 8)
    #width           = 32 #trial.suggest_int("width", 16, 32, step=4) 

    #  hardcoded hyperparameters
    lr_start        = 0.0006
    dr              = 0.9120
    wd              = 0.0052
    p_do            = 0.08
    modes           = 4 
    width           = 32
    num_layers      = 6

    batch_size = 8
    num_epochs = 40
    n_samples_train = inputs_train.shape[0]
    n_batches_train = jnp.ceil((n_samples_train/batch_size)).astype(int) 
    n_samples_validation = inputs_validation.shape[0]
    n_batches_validation = jnp.ceil((n_samples_validation/batch_size)).astype(int) 
    n_samples_test = inputs_test.shape[0]
    n_batches_test = jnp.ceil((n_samples_test/batch_size)).astype(int)

    
    # initialize the model
    model = UFNO3d(in_channels=3, out_channels=1, num_layers=num_layers, modes_x=modes, modes_y=modes, modes_z=modes, width= width, p_do=p_do, key=jax.random.PRNGKey(0))
    # load saved model uncomment the following code (if you want to do this, in the previous line before model needs to be initialized with the same hyperparameters (modes, width, num_layers) as the saved model) 
    #model = eqx.tree_deserialise_leaves("surrogate_models/final_ufno_3d_time.eqx", model) # you should uncomment the training when using a saved model

    # initialization of the optimizer (including lr schedule)
    schedule = optax.exponential_decay(lr_start, n_batches_train* num_epochs, dr)
    optim = optax.adamw(learning_rate=schedule, weight_decay=wd)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    print("Model and Optimizer initialized")

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

    # different loss functions
    def mse_loss(model, x, y, key=None, deterministic=False):
        y_pred = jax.vmap(model, in_axes=(0, None, None))(x, key, deterministic)  
        return jnp.mean(jnp.square(y_pred - y))
    
    def relative_loss(model, x, y, key=None, deterministic=False):
        y_pred = jax.vmap(model, in_axes=(0, None, None))(x, key, deterministic)  
        y_pred_flat = y_pred.reshape((-1))
        y_flat = y.reshape((-1))
        rel_err = jnp.abs((y_flat - y_pred_flat)/(y_flat))
        mask = (y_flat != 0) # ignoring points where y is exactly zero to avoid division by zero
        rel_err_masked = rel_err * mask
        mean_error = jnp.sum(rel_err_masked) / jnp.sum(mask) # mean relative error over non-zero points
        return mean_error
        
    def ufno_loss(model, x, y, key=None, deterministic=False):
        y_pred = jax.vmap(model, in_axes=(0, None, None))(x, key, deterministic)  
        original_loss =  jnp.mean(jnp.square(y_pred - y))  
        dy_pred_x, dy_pred_y, dy_pred_z = jnp.gradient(y_pred, (1/(x.shape[-3]-1)), (1/(x.shape[-2]-1)), (1/(x.shape[-1]-1)), axis=(-3,-2,-1))
        dy_x, dy_y, dy_z = jnp.gradient(y, (1/(x.shape[-3]-1)), (1/(x.shape[-2]-1)), (1/(x.shape[-1]-1)), axis=(-3,-2,-1))
        dy_pred_x = dy_pred_x[:, :, 1:-1, 1:-1, 1:-1]  
        dy_pred_y = dy_pred_y[:, :, 1:-1, 1:-1, 1:-1] 
        dy_pred_z = dy_pred_z[:, :, 1:-1, 1:-1, 1:-1] 
        dy_pred = jnp.concatenate((dy_pred_x, dy_pred_y, dy_pred_z), axis=-4) 
        dy_x = dy_x[:, :, 1:-1, 1:-1, 1:-1] 
        dy_y = dy_y[:, :, 1:-1, 1:-1, 1:-1]
        dy_z = dy_z[:, :, 1:-1, 1:-1, 1:-1]
        dy= jnp.concatenate((dy_x, dy_y, dy_z), axis=-4) 
        gradient_loss = jnp.mean(jnp.square(dy_pred - dy))  
        return original_loss + 0.5*gradient_loss
    
    def ufno_loss_2(model, x, y, key=None, deterministic=False): 
        num_examples = x.shape[0]
        y_pred = jax.vmap(model, in_axes=(0, None, None))(x, key, deterministic)
        y_pred_flat = y_pred.reshape((num_examples, -1))
        y_flat = y.reshape((num_examples, -1))
        diff_norms = jnp.linalg.norm(y_pred_flat - y_flat, ord=2, axis=1)
        y_norms    = jnp.linalg.norm(y_flat, ord=2, axis=1)
        dy_pred_x, dy_pred_y, dy_pred_z = jnp.gradient(y_pred, (1/(x.shape[-3]-1)), (1/(x.shape[-2]-1)), (1/(x.shape[-1]-1)), axis=(-3,-2,-1))
        dy_x, dy_y, dy_z = jnp.gradient(y, (1/(x.shape[-3]-1)), (1/(x.shape[-2]-1)), (1/(x.shape[-1]-1)), axis=(-3,-2,-1))
        dy_pred_x = dy_pred_x[:, :, 1:-1, 1:-1, 1:-1]  
        dy_pred_y = dy_pred_y[:, :, 1:-1, 1:-1, 1:-1] 
        dy_pred_z = dy_pred_z[:, :, 1:-1, 1:-1, 1:-1] 
        dy_pred = jnp.concatenate((dy_pred_x, dy_pred_y, dy_pred_z), axis=-4) 
        dy_x = dy_x[:, :, 1:-1, 1:-1, 1:-1] 
        dy_y = dy_y[:, :, 1:-1, 1:-1, 1:-1]
        dy_z = dy_z[:, :, 1:-1, 1:-1, 1:-1]
        dy= jnp.concatenate((dy_x, dy_y, dy_z), axis=-4) 
        dy_pred_flat = dy_pred.reshape((num_examples, -1))
        dy_flat = dy.reshape((num_examples, -1))
        graddiff_norms = jnp.linalg.norm(dy_pred_flat - dy_flat, ord=2, axis=1)
        dy_norms    = jnp.linalg.norm(dy_flat, ord=2, axis=1)
        return jnp.mean(diff_norms/y_norms) + 0.5*jnp.mean(graddiff_norms/dy_norms)      

    # definition of training and evaluation steps
    @eqx.filter_jit(donate="all")
    def train_step(model, opt_state, key, x, y, sharding):
        replicated = sharding.replicate()
        model, opt_state = eqx.filter_shard((model, opt_state), replicated)
        x, y = eqx.filter_shard((x, y), sharding)
        loss, grads = eqx.filter_value_and_grad(ufno_loss_2)(model, x, y, key, deterministic=False)  
        updates, opt_state = optim.update(grads, opt_state, model)  
        model = eqx.apply_updates(model, updates)
        model, opt_state = eqx.filter_shard((model, opt_state), replicated)

        return model, opt_state, loss
    
    @eqx.filter_jit(donate="all-except-first")
    def evaluate(model, x, y, sharding):
        replicated = sharding.replicate()
        model = eqx.filter_shard(model, replicated)
        x, y = eqx.filter_shard((x, y), sharding)
        loss = mse_loss(model, x, y, deterministic=True)  
        return loss
    
    @eqx.filter_jit(donate="all-except-first")
    def evaluate_in_test(model, x, y, sharding):
        replicated = sharding.replicate()
        model = eqx.filter_shard(model, replicated)
        x, y = eqx.filter_shard((x, y), sharding)
        loss = relative_loss(model, x, y, deterministic=True)  
        return loss

    model = eqx.filter_shard(model, replicated) 
    loss_history = []
    val_loss_history = []
    shuffle_key = jax.random.PRNGKey(20)
    
    print("Starting training...")
    
    # training (and validation) 
    for e in range(num_epochs):
        shuffle_key, perm_key = random.split(shuffle_key)
        perm = jax.random.permutation(perm_key, inputs_train.shape[0])
        train_x_set = inputs_train[perm]
        train_y_set = outputs_train[perm]
        for i in range(0, train_x_set.shape[0], batch_size):
            if i + batch_size > inputs_train.shape[0]: # skip last batch so sharding works
                break
            batch_x = train_x_set[i:i + batch_size]
            batch_y = train_y_set[i:i + batch_size]
            shuffle_key, subkey = jax.random.split(shuffle_key)
            batch_x, batch_y = eqx.filter_shard((batch_x, batch_y), sharding)
            model, opt_state, loss = train_step(model, opt_state, subkey, batch_x, batch_y, sharding)
            loss_history.append(loss)
    
        val_loss_tracker = 0
        for i in range(0, inputs_validation.shape[0], batch_size):
            if i + batch_size > inputs_validation.shape[0]: # skip last batch so sharding works
                break
            batch_x = inputs_validation[i:i + batch_size]
            batch_y = outputs_validation[i:i + batch_size]
         
            batch_x, batch_y = eqx.filter_shard((batch_x, batch_y), sharding)
            val_loss = evaluate(model, batch_x, batch_y, sharding)
            val_loss_tracker += val_loss
        val_loss_history.append(val_loss_tracker/(n_batches_test-1))
        print(f"Epoch {e+1}/{num_epochs}, Validation Loss (MSE): {val_loss_history[-1]:.5f}, Training Loss (Custom Loss 2): {np.mean(loss_history[-n_batches_train:-1]):.5f}")
    
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
    
    # save best model - comment this in if you want to save your best model
    #if not hasattr(objective, "best_loss") or test_loss < objective.best_loss:
    #    objective.best_loss = test_loss
    #    eqx.tree_serialise_leaves("surrogate_models/custom_ufno_3d_time.eqx", model)
    
    return test_loss
    
def preprocess_data(data_x, data_y, xp_min, xp_max, yp_min, yp_max):
    if xp_min.shape == ():
        pass
    else:
        xp_min = xp_min[:, None, None, None]
        xp_max = xp_max[:, None, None, None]
    if yp_min.shape == ():
        pass
    else:
        yp_min = yp_min[:, None, None, None]
        yp_max = yp_max[:, None, None, None]

    data_x = (np.log10(data_x + 1e-8) - xp_min)/ (xp_max - xp_min)  
    data_y = (np.log10(data_y + 1e-8) - yp_min)/ (yp_max - yp_min)
    return data_x, data_y


def unpreprocess_data(data_x, data_y, xp_min, xp_max, yp_min, yp_max):
    if xp_min.shape == ():
        pass
    else:
        xp_min = xp_min[:, None, None, None]
        xp_max = xp_max[:, None, None, None]
    if yp_min.shape == ():
        pass
    else:
        yp_min = yp_min[:, None, None, None]
        yp_max = yp_max[:, None, None, None]

    data_x = (data_x * (xp_max - xp_min)) + xp_min
    data_y = (data_y * (yp_max - yp_min)) + yp_min
    return data_x, data_y



   
if __name__ == "__main__":
    print("Starting...")

    # in the README file there is a description on how to download the datsets

    inputs_train = np.load('datasets/inputs_train_3d_time.npy')#, mmap_mode='r')
    outputs_train = np.load('datasets/outputs_train_3d_time.npy')#, mmap_mode='r')
    inputs_validation = np.load('datasets/inputs_validation_3d_time.npy')#, mmap_mode='r')
    outputs_validation = np.load('datasets/outputs_validation_3d_time.npy')#, mmap_mode='r')
    inputs_test = np.load('datasets/inputs_test_3d_time.npy')#, mmap_mode='r')
    outputs_test = np.load('datasets/outputs_test_3d_time.npy')#, mmap_mode='r')

    # values with which data was preprocessed
    xp_min = jnp.array([-1.7541726, -8.000001, -8.000001]) 
    xp_max = jnp.array([0.81056046, 4.197271, 5.4913197]) 
    yp_min = jnp.array([-8.000001]) 
    yp_max = jnp.array([5.4913197])

    print("Data has been loaded")

    # start the Optuna study (you can adjust the number of trials) and print out best parameters
    time1 = time.time()
    study = optuna.create_study(direction="minimize", study_name="3d-time-study")
    study.optimize(objective, n_trials=1, n_jobs=1, gc_after_trial=True, show_progress_bar=True)
    print("Best parameters:", study.best_params)
    print("Validation loss of best parameters:", study.best_value)
    time2 = time.time()
    print(f"Total time for optimization: {time2-time1:.4f} seconds")


    
    
    
    
    
    
    
    

    