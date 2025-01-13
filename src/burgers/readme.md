### Run PINN to solve Burgers' Equation in one spatial dimension and time:

adaptive.py - Starting with a specified initial distribution, this code then uses the specified guiding information (input1) to guide adaptive resampling of collocation points for a specified number of loops.

fixed_distribution.py - Runs once without resampling with a specified, fixed, initial distribution.

adaptive_IC_\*.py - These solve the same PDE, but with variations to the initial condition.

adaptive_v_\*.py - These use the original initial condition, but vary the value of v in the PDE.


### Use FDM to obtain a ground truth solution to compare against with PINNs.

fdm_main_v01.py - v=0.01

fdm_main_v001.py - v=0.001

The two provided use a viscosity of 0.01, or 0.001, but this can be easily changed. Note the value used in the PINNs examples is v=0.01/pi.
