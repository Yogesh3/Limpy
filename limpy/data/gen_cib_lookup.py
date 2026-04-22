import numpy as np
import time
from limpy import cib
from scipy.optimize import fsolve
import debugpy
import argparse
from myfuncs import mpi as ympi

#Parser
parser = argparse.ArgumentParser(description='Calculates lookup table and interpolation functions for Shang model CIB SED')
parser.add_argument('--params', default= 'planck', help= 'Either Planck13 or Viero19 CIB parameters', choices=['planck, viero'])
args = parser.parse_args()


#Set CIB Params
cib_params = {}
if args.params.lower() == 'planck':
    cib_params['alpha'] = 0.36
    cib_params['beta'] = 1.75
    cib_params['gamma'] = 1.7
    cib_params['delta'] = 3.6
    cib_params['Td_o'] = 24.4
    cib_params['logM_eff'] = 12.6
    cib_params['var'] = 0.5
    # cib_params['L_o'] = 6.4e-8      # Jy * Mpc^2 / M_sun / Hz
    cib_params['L_o'] = 1.59e-15       # L_sol / M_sol / Hz
elif args.params.lower() == 'viero':
    cib_params['alpha'] = 0.2
    cib_params['beta'] = 1.6
    cib_params['gamma'] = 1.7
    cib_params['delta'] = 2.4
    cib_params['Td_o'] = 20.7
    cib_params['logM_eff'] = 12.3
    cib_params['var'] = 0.5
    # cib_params['L_o'] = 6.4e-8      # Jy * Mpc^2 / M_sun / Hz
    cib_params['L_o'] = 1.59e-15       # L_sol / M_sol / Hz
else:
    raise ValueError("Don't recognize the name of that set of CIB parameter constraints")


#Exact Redshifts
zs_all = np.linspace(0, 20, int(6e5))

#MPI
comm, current_rank, zs_this_thread = ympi.distMPI(zs_all)

# #Debugging
# port = 5678 + current_rank
# debugpy.listen(("localhost", port))
# ympi.mpiprint(f"Rank {current_rank} waiting for debugger on port {port}...")
# debugpy.wait_for_client()

start_time = time.time()

#Calculate SED Parameters
temp_array = cib_params['Td_o'] * (1 + zs_this_thread)**cib_params['alpha']
nu_o_guess = np.ones(temp_array.shape, dtype=np.float64) * 5.0e12   # initial guess
A_guess = np.ones(temp_array.shape, dtype=np.float64) * 1.0e32      # initial guess
debugpy.breakpoint()
sol = fsolve(cib.sysEquations, np.concatenate((nu_o_guess, A_guess)), args=(temp_array, cib_params['beta'], cib_params['gamma']))
nu_o_array = sol[:temp_array.size] 
A_array = sol[temp_array.size:] 

#Restructure for Gatherv
nu_o_array = np.array(nu_o_array)[:, None]
A_array = np.array(A_array)[:, None]

comm.Barrier()

#Collect Table
nu_o_all = ympi.Gatherv(nu_o_array, len(zs_all))
A_all = ympi.Gatherv(A_array, len(zs_all))


if current_rank == 0:
    ympi.printTotalTime(start_time, time.time(), Nthings= len(zs_all))

    #Save Data
    arrays_to_save = {}
    arrays_to_save['z'] = zs_all
    arrays_to_save['nu_o'] = nu_o_all
    arrays_to_save['A'] = A_all
    np.savez('/scratch/ymehta3/cib_lookup_table.npz',
            **arrays_to_save
            )