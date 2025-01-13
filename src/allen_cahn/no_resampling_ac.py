"""Run PINN to solve Allen Cahn equation. Only use D=E-3
no_resampling_ac.py - No resample, 400k iterations. Saves loss to check convergence.

Usage:
    no_resampling_ac.py [--N=<NumDomain>] [--IM=<InitialMethod>] [--DEP=<depth>]
    no_resampling_ac.py -h | --help
Options:
    -h --help                   Display this help message
    --N=<NumDomain>             Number of collocation points for training [default: 2000]
    --IM=<InitialMethod>        Initial distribution method from: "Grid","Random","LHS", "Halton", "Hammersley", "Sobol" [default: Random]
    --DEP=<depth>               Depth of the network [default: 3]
"""
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Initial imports and some function definitions.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import os
from docopt import docopt
import skopt
from distutils.version import LooseVersion
import deepxde as dde
from deepxde.backend import tf
import numpy as np
from scipy.io import loadmat
import time

os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
dde.config.set_default_float("float64")
dde.optimizers.config.set_LBFGS_options(maxiter=385000)

def gen_testdata_E3(): # This function opens the ground truth solution. Commented out directory is for local
    data = loadmat("usol_D_0.001_k_5.mat") # data = loadmat("src/allen_cahn/usol_D_0.001_k_5.mat")
    t = data["t"]
    x = data["x"]
    u = data["u"]
    dt = dx = 0.01
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]
    return X, y

def quasirandom(n_samples, sampler): # This function creates pseudorandom distributions if initial method is specified.
    space = [(-1.0, 1.0), (0.0, 1.0)]
    if sampler == "LHS":
        sampler = skopt.sampler.Lhs(
            lhs_type="centered", criterion="maximin", iterations=1000
        )
    elif sampler == "Halton":
        sampler = skopt.sampler.Halton(min_skip=-1, max_skip=-1)
    elif sampler == "Hammersley":
        sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
    elif sampler == "Sobol":
        # Remove the first point [0, 0, ...] and the second point [0.5, 0.5, ...], which
        # are too special and may cause some error.
        if LooseVersion(skopt.__version__) < LooseVersion("0.9"):
            sampler = skopt.sampler.Sobol(min_skip=2, max_skip=2, randomize=False)
        else:
            sampler = skopt.sampler.Sobol(skip=0, randomize=False)
            return np.array(sampler.generate(space, n_samples + 2)[2:])
    return np.array(sampler.generate(space, n_samples))

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main code start
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main(NumDomain=2000, method="Random", depth=3): # Main Code
    print(f"NumDomain equals {NumDomain}")
    print(f"Method equals {method}")
    print(f"Depth equals {depth}")
    start_t = time.time() #Start time.

    def pde(x, y): # Define Allen Cahn Equation
        u = y
        du_xx = dde.grad.hessian(y, x, i=0, j=0)
        du_t = dde.grad.jacobian(y, x, i=0, j=1)
        return du_t - 0.001 * du_xx + 5 * (u**3 - u)

    X_test, y_true = gen_testdata_E3()
    # This chunk of code describes the problem using dde structure. Varies depending on prescribed initial distribution.
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    if method == "Grid":
        data = dde.data.TimePDE(
            geomtime, pde, [], num_domain=NumDomain, num_test=10000, train_distribution="uniform"
        )
    elif method == "Random":
        data = dde.data.TimePDE(
            geomtime, pde, [], num_domain=NumDomain, num_test=10000, train_distribution="pseudo"
        )
    elif method in ["LHS", "Halton", "Hammersley", "Sobol"]:
        sample_pts = quasirandom(NumDomain, method)
        data = dde.data.TimePDE(
            geomtime,
            pde,
            [],
            num_domain=0,
            num_test=10000,
            train_distribution="uniform",
            anchors=sample_pts,
        )

    net = dde.maps.FNN([2] + [64] * depth + [1], "tanh", "Glorot normal") # This defines the NN layers, their size and activation functions.

    def output_transform(x, y):
        x_in = x[:,0:1]
        t_in = x[:,1:2]
        return t_in * (1 + x_in) * (1 - x_in) * y + tf.square(x_in) * tf.cos(np.pi * x_in)
    net.apply_output_transform(output_transform)
    
    # The only training as no resampling
    model = dde.Model(data, net)
    print("Adam steps")
    model.compile("adam", lr=0.001)
    model.train(epochs=15000, display_every=10000)
    print("Initial L-BFGS steps")
    model.compile("L-BFGS")
    losshistory, train_state = model.train(display_every=10000)
    
    y_pred = model.predict(X_test)
    error_final = dde.metrics.l2_relative_error(y_true, y_pred)
    time_taken = (time.time()-start_t)
    print("Time taken:", time_taken)
    
    dde.saveplot(losshistory, train_state, issave=True, isplot=False, 
                 loss_fname=f"allencahn_no_replacement_D{depth}_{method}_N{NumDomain}_loss_info.dat", 
                 train_fname=f"allencahn_no_replacement_D{depth}_{method}_N{NumDomain}_finalpoints.dat", 
                 test_fname=f"allencahn_no_replacement_D{depth}_{method}_N{NumDomain}_finalypred.dat",
                 output_dir="../results/additional_info")
    return error_final, time_taken
 
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calling main and saving results
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = docopt(__doc__)
    NumDomain=int(args['--N'])
    method=str(args['--IM'])
    depth=int(args["--DEP"])

    error_final, time_taken = main(NumDomain=NumDomain,method=method, depth=depth) # Run main, record error history and final accuracy.

    if np.isscalar(time_taken):
        time_taken = np.atleast_1d(time_taken)
    if np.isscalar(error_final):
        error_final = np.atleast_1d(error_final)
    
    output_dir = "../results/performance_results/allen_cahn"  # Replace with your desired output directory path
    error_final_fname = f"allencahn_no_resample_D{depth}_{method}_N{NumDomain}_error_final.txt"
    time_taken_fname = f"allencahn_no_resample_D{depth}_{method}_N{NumDomain}_time_taken.txt"
    
    # If results directory does not exist, create it
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Define the full file paths
    error_final_fname = os.path.join(output_dir, error_final_fname)
    time_taken_fname = os.path.join(output_dir, time_taken_fname)
    
    # Define function to append to file. The try/exception was to ensure when ran as task array that saving won't fail in the rare case that
    # the file is locked for saving by a different job.
    def append_to_file(file_path, data):
        try:    
            with open(file_path, 'ab') as file:
                np.savetxt(file,data)
                print(f"Saved to \"{output_dir}\" directory successfully")
        except Exception as e:
            print(f"An exception occurred: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            try:
                with open(file_path, 'ab') as file:
                    np.savetxt(file, data)
                    print(f"Saved to \"{output_dir}\" directory successfully")
            except Exception as e2:
                print(f"An exception occurred again: {e2}")

    # Use function to append to file.
    append_to_file(error_final_fname, error_final)
    append_to_file(time_taken_fname, time_taken)