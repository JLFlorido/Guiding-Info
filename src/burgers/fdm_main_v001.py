"""
fdm_main_v001.py
This python script uses scipy#s odeint to solve Burger's PDE via taking finite differences, for v=0.001
Usage:
    fdm_main_v001.py [--N=<N>] [--nt=<nt>]
    fdm_main_v001.py -h | --help
Options:
    -h --help       Display this help message
    --N=<N>         Number of points in x
    --nt<nt>        Number of time divisions
"""
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import time
from docopt import docopt

args = docopt(__doc__)
N=int(args['--N'])
nt=int(args['--nt'])

dx = 2 / N
b = 0.001  # viscosity term

# Define Function to pass to odeint
def burgers(u_int, t, b, N):
    u_local = np.zeros([N + 1])
    u_local[1:N] = u_int
    dudt = np.zeros([N - 1])

    for i in range(1, N):
        dudt[i - 1] = b * (u_local[i + 1] - 2 * u_local[i] + u_local[i - 1]) / (
            dx * dx
        ) - u_local[i] * (u_local[i + 1] - u_local[i - 1]) / (
            2 * dx
        )
    return dudt


x0 = np.linspace(-1, 1, N + 1) 
u0 = -np.sin(np.pi*x0)
# Interior u0 points
u0_int = u0[1:N]

# Define Time Interval
t = np.linspace(0, 1, nt)

print("Begin Calculation...\n")
start_time = time.time()

# Where the solution is computed
u_int = odeint(burgers, u0, t, args=(b,N+2))
# u_int = odeint(burgers, u0_int, t, args=(b, N))
u_int = np.transpose(u_int)
# Writing BC into array
## u[:, 0] = u0
u = np.zeros([N + 1, nt])
u[0, :] = 0
u[N, :] = 0
u[0:N+1, :] = u_int
print(t.shape)
print(x0.shape)
print(u.shape)

#-------------------------------------------------------------------------------
# Print time taken for main code to run
print("--- %s seconds ---" % (time.time() - start_time))
print(f"To check, v = {b}")

# Export Data
np.savez(f"Burgers_v{b}_N{N}_nt{nt}.npz", t=t, x=x0, exact=u)
print("\n --- Data Exported ---")