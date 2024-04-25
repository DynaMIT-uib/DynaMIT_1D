import numpy as np
from diffmatrix import Diffmatrix
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import xarray as xr
from cases import cases

#constants
mu0 = 4 * np.pi * 1e-7 # vacuum permeability
e = 1.6e-19 # elemenatry charge


for case in cases:

    for key, val in case.items(): # unpack the dictionary thing
        exec(key + '=val')

    print('working on ' + filename)

    #grid
    x = np.linspace(0, L, N) # grid
    dx = x[1] - x[0]

    # initial conditions
    ki = e * B0 / (nu * mi)
    P0 = np.ones_like(x) * alpha * (B0 * Sigma_P / e * (1 + ki**2) / ki)**2 #np.ones_like(x) * 1e15 * alpha # plasma production rate
    P = P0
    vix = np.zeros(N) # ion velocity in horizontal plane x direction
    viy = np.zeros(N) # ion velocity in horizontal plane y direction
    vex = np.zeros(N) # electron velocity in horizontal plane x direction
    vey = np.zeros(N) # electron velocity in horizontal plane y direction
    v0 = v0_max * np.exp(-((x-L//2)/(200e3))**(2 * 3)) # imposed flow
    n = np.full(N, np.sqrt(P0/alpha)) # particles per square meter

    # initial magnetic field line positions:
    s_y = np.zeros((2, x.size))
    s_x = np.zeros((2, x.size))

    # define Knight relation logistic function
    ju0 = 1e-6 # where the transition from 0 to 1 is centered
    knight = lambda ju: (np.sign(ju) + 1) / 2 * ju#1 / (1 + np.exp(-(ju - ju0) * 8e6))
    knight_scale = 1e-2 * P0[0] # Prodution rate is knight_cale * ju where knight(ju) is 1

    # model arrays (not meant to be modified)
    By_above = np.full(N, 0) # horizontal magnetic field in x direction above current sheet
    Bx_above = np.full(N, 0) # horizontal magnetic field in x direction above current sheet
    Bx_below = np.full(N, 0) # horizontal magnetic field in x direction below current sheet
    By_below = np.full(N, 0) # horizontal magnetic field in y direction below current sheet
    jhy      = np.full(N, 0) # horizontal magnetic field in y direction below current sheet
    jhy_dB   = np.full(N, 0) # horizontal magnetic field in y direction below current sheet


    Br = np.full(N, B0)

    # get matrix for differentiating in x direction
    DX  = Diffmatrix(3, N, 1, order = 1, h = dx).D

    # make G matrix for solving Laplace equation
    print('preparing Laplace solver')
    nn = np.arange(1, N).reshape((1, -1))
    xc = x.reshape((-1, 1)) # column vector
    Gr =  nn * np.pi / L * np.cos(nn * np.pi * xc / L)
    Gx =  -nn * np.pi / L * np.sin(nn * np.pi * xc / L)
    Grinv = np.linalg.pinv(Gr, rcond = 0)

    Gr_ground =  Gr * np.exp(-nn * np.pi / L * r0)
    Gx_ground =  Gx * np.exp(-nn * np.pi / L * r0) 

    # matrices for relating line currents to magnetic field:
    B_lc_ground = mu0 * dx / (2 * np.pi * np.sqrt((xc - xc.T + dx/2)**2 + r0**2))
    B_lc = mu0 * dx / (2 * np.pi * (xc.T - dx/2 - xc)) # B in ionosphere
    B_lc_inv = np.linalg.pinv(B_lc)
    theta = np.arctan((xc.T - dx/2 - xc) / r0)
    Bx_lc_ground = B_lc_ground * np.cos(theta)
    Br_lc_ground = B_lc_ground * np.sin(theta)




    print('starting simulation')
    counter, t = 0, 0
    keys = ['vix', 'viy', 'Br', 'n', 'Ey', 'Ex', 'By+', 'Bx+', 'Bx-', 't', 'jhx', 'jhy', 'jr', 'Br_ground', 'Bx_ground', 'B_lc_ground', 'Bx_lc_ground', 'Br_lc_ground', 'jy_lc']
    data = {k:[] for k in keys}
    vmagnetosphere = v0

    while True:
        s_x = s_x - s_x[0] # subtract the bottom position of the field lines
        s_y = s_y - s_y[0] # subtract the bottom position of the field lines
        s_y[1] += vmagnetosphere * dt
        s_y[0] += vey * dt
        s_x[0] += vex * dt

        slope_y = (s_y[1] - s_y[0])/height
        By_above = B0 * slope_y / np.sqrt(slope_y**2 + 1)
        jhx = -By_above / mu0

        if np.any(np.isnan(jhy)):
            print(t)
            break

        #slope_x = (s_x[1] - s_x[0])/height
        #Bx_above = B0 * slope_x / np.sqrt(slope_x**2 + 1)
        jhy = -2*Bx_below / mu0 

        vex = vix - jhx / (n * e)
        vey = viy - jhy / (n * e) 

        Ex = -vey * Br
        Ey =  vex * Br



        # propagate the time dependent quantities:

        # density change:
        dn = DX.dot(vix * n) + P - n**2 * alpha

        # momentum change:
        dnvix = (Ex + viy * Br) * n * e / mi - n * nu * vix
        dnviy = (Ey - vix * Br) * n * e / mi - n * nu * viy

        # update density:
        n_new = n + dn * dt

        # update velocity:
        vix = n / n_new * vix + dnvix * dt / n_new
        viy = n / n_new * viy + dnviy * dt / n_new

        n = n_new


        # magnetic field:
        dBr = -DX.dot(Ey)
        Br = Br + dBr * dt


        h = Grinv.dot(Br - B0)
        Bx_below = Gx.dot(h)
        Bx_above = -Bx_below

        jy_lc = B_lc_inv.dot(Br - B0) # save this just for checking the consistency with Laplace




        if (counter % STEP == 0) or ((counter < STEP) & (counter % 5 == 0)):
            data['vix'].append(vix)
            data['viy'].append(viy)
            data['n'].append(n)
            data['Br'].append(Br)
            data['Ey'].append(Ey)
            data['Ex'].append(Ex)
            data['By+'].append(By_above)
            data['Bx+'].append(Bx_above)
            data['Bx-'].append(Bx_below)
            data['jhy'].append(jhy)
            data['jhx'].append(jhx)
            data['t'].append(t)
            data['jr'].append(DX.dot(By_above) / mu0)
            data['Br_ground'].append(Gr_ground.dot(h))
            data['Bx_ground'].append(Gx_ground.dot(h))
            data['B_lc_ground'].append(B_lc_ground.dot(jhy))
            data['Bx_lc_ground'].append(Bx_lc_ground.dot(jhy))
            data['Br_lc_ground'].append(Br_lc_ground.dot(jhy))
            data['jy_lc'].append(jy_lc)
            Brm = Gr.dot(h)
            print(t, np.linalg.norm(Brm - (Br - B0)) / np.linalg.norm(Br - B0), vix.max(), vex.min(), viy.max(), vey.max())

        t = t + dt
        counter += 1

        if t >= t_end:
            break


    for key in keys:
        if key != 't':
            data[key] = np.vstack(data[key])
        else:
            data[key] = np.array(data[key])


    ddict = {}
    for k in [key for key in keys if key != 't']:
        ddict[k]= (['time', 'x'], data[k])


    coords = {'time': data['t'], 'x':x}

    ds = xr.Dataset(ddict, coords)

    ds['B0'] = B0
    ds['r0'] = r0
    ds['height'] = height
    ds['v0'] = v0
    ds['P0'] = P0
    ds['mi'] = mi
    ds['nu'] = nu
    ds['Sigma_P'] = Sigma_P
    ds['Sigma_H'] = Sigma_P / ki

    dB = np.sqrt((ds['Br_ground'] - ds['Br_lc_ground'])**2 + (ds['Bx_ground'] - ds['Bx_lc_ground'])**2)


    ds.to_netcdf(filename + '.netcdf')




