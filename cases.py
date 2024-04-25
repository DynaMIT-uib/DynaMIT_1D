import numpy as np

e = 1.6e-19 # elemenatry charge

# principle variables:
B0 = -50000.e-9 # main magnetic field in vertical direction
r0 = 100e3 # height of ionosphere
mi = 16 * 1.66e-27 # ion mass in kg
dt = 0.0001 # simulation time step
t_end = 120

cases = [
dict(filename = './output/Sigma_H_10_ratio_0.5',
     mi       = mi, # ion mass in kg
     v0_max   = 500, # m/s
     B0       = B0, # main magnetic field in vertical direction
     r0       = r0, # height of ionosphere
     alpha    = 1e-13, # recombination coefficient [m^3/s] (https://doi.org/10.1029/RS007i004p00469)
     nu       = 0.5 * e * np.abs(B0) / mi,  # ion neutral collision frequency (Hall/Pedersen = nu * mi / (e * B0))  
     height   = 5 * r0, # distance from ionosphere to magnetospheric driver
     Sigma_P  = 20, # Pedersen conductance (used to scale the production rate)
     L        = 2000e3, # length of simulation domain [m]
     N        = 1000, # number of grid cells in x direction
     dt       = dt, # simulation time step
     STEP     = int(np.round(0.15/dt)), # save output every STEP timestep
     t_end    = t_end), # stop simulation after this many seconds

dict(filename = './output/Sigma_H_10_ratio_1',
     mi       = mi, # ion mass in kg
     v0_max   = 500, # m/s
     B0       = B0, # main magnetic field in vertical direction
     r0       = r0, # height of ionosphere
     alpha    = 1e-13, # recombination coefficient [m^3/s] (https://doi.org/10.1029/RS007i004p00469)
     nu       = 1 * e * np.abs(B0)/mi,  # ion neutral collision frequency (Hall/Pedersen = nu * mi / (e * B0))  
     height   = 5 * r0, # distance from ionosphere to magnetospheric driver
     Sigma_P  = 10, # Pedersen conductance (used to scale the production rate)
     L        = 2000e3, # length of simulation domain [m]
     N        = 1000, # number of grid cells in x direction
     dt       = dt, # simulation time step
     STEP     = int(np.round(0.15/dt)), # save output every STEP timestep
     t_end    = t_end), # stop simulation after this many seconds

dict(filename = './output/Sigma_H_10_ratio_5',
     mi       = mi, # ion mass in kg
     v0_max   = 500, # m/s
     B0       = B0, # main magnetic field in vertical direction
     r0       = r0, # height of ionosphere
     alpha    = 1e-13, # recombination coefficient [m^3/s] (https://doi.org/10.1029/RS007i004p00469)
     nu       = 5 * e * np.abs(B0)/mi,  # ion neutral collision frequency (Hall/Pedersen = nu * mi / (e * B0))  
     height   = 5 * r0, # distance from ionosphere to magnetospheric driver
     Sigma_P  = 2, # Pedersen conductance (used to scale the production rate)
     L        = 2000e3, # length of simulation domain [m]
     N        = 1000, # number of grid cells in x direction
     dt       = dt, # simulation time step
     STEP     = int(np.round(0.15/dt)), # save output every STEP timestep
     t_end    = t_end), # stop simulation after this many seconds

dict(filename = './output/Sigma_H_5_ratio_0.5',
     mi       = mi, # ion mass in kg
     v0_max   = 500, # m/s
     B0       = B0, # main magnetic field in vertical direction
     r0       = r0, # height of ionosphere
     alpha    = 1e-13, # recombination coefficient [m^3/s] (https://doi.org/10.1029/RS007i004p00469)
     nu       = 0.5 * e * np.abs(B0) / mi,  # ion neutral collision frequency (Hall/Pedersen = nu * mi / (e * B0))  
     height   = 5 * r0, # distance from ionosphere to magnetospheric driver
     Sigma_P  = 10, # Pedersen conductance (used to scale the production rate)
     L        = 2000e3, # length of simulation domain [m]
     N        = 1000, # number of grid cells in x direction
     dt       = dt, # simulation time step
     STEP     = int(np.round(0.15/dt)), # save output every STEP timestep
     t_end    = t_end), # stop simulation after this many seconds

dict(filename = './output/Sigma_H_5_ratio_1',
     mi       = mi, # ion mass in kg
     v0_max   = 500, # m/s
     B0       = B0, # main magnetic field in vertical direction
     r0       = r0, # height of ionosphere
     alpha    = 1e-13, # recombination coefficient [m^3/s] (https://doi.org/10.1029/RS007i004p00469)
     nu       = 1 * e * np.abs(B0)/mi,  # ion neutral collision frequency (Hall/Pedersen = nu * mi / (e * B0))  
     height   = 5 * r0, # distance from ionosphere to magnetospheric driver
     Sigma_P  = 5, # Pedersen conductance (used to scale the production rate)
     L        = 2000e3, # length of simulation domain [m]
     N        = 1000, # number of grid cells in x direction
     dt       = dt, # simulation time step
     STEP     = int(np.round(0.15/dt)), # save output every STEP timestep
     t_end    = t_end), # stop simulation after this many seconds

dict(filename = './output/Sigma_H_5_ratio_5',
     mi       = mi, # ion mass in kg
     v0_max   = 500, # m/s
     B0       = B0, # main magnetic field in vertical direction
     r0       = r0, # height of ionosphere
     alpha    = 1e-13, # recombination coefficient [m^3/s] (https://doi.org/10.1029/RS007i004p00469)
     nu       = 5 * e * np.abs(B0)/mi,  # ion neutral collision frequency (Hall/Pedersen = nu * mi / (e * B0))  
     height   = 5 * r0, # distance from ionosphere to magnetospheric driver
     Sigma_P  = 1, # Pedersen conductance (used to scale the production rate)
     L        = 2000e3, # length of simulation domain [m]
     N        = 1000, # number of grid cells in x direction
     dt       = dt, # simulation time step
     STEP     = int(np.round(0.15/dt)), # save output every STEP timestep
     t_end    = t_end), # stop simulation after this many seconds
]
