""" 

Create cartoon figure to describe the sequence of events to change the magnetic field. Uses dbdt.py model output

"""

import numpy as np
from diffmatrix import Diffmatrix
import matplotlib.pyplot as plt
from scipy.integrate import RK45
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from cases import cases

plt.xkcd()

three_figs = True # set to True if you want subplot labels appropriate for three different figures

fns  = [case['filename'] + '.netcdf' for case in cases]

e = 1.6e-19 # elemenatry charge
mu0 = 4 * np.pi * 1e-7 # vacuum permeability

# load cdf and extract simulation parameters
data = xr.open_dataset(fns[-2])
v0 = data['v0'].values
n0 = data['n' ].values[0, 0]
xx = data['x'].values
B0 = data['B0'].values
r0 = data['r0'].values
L = xx[-1] - xx[0]
N  = len(xx)
v0_max = v0.max()


# make G matrix for solving Laplace equation
print('preparing Laplace solver')
nn = np.arange(1, N//2).reshape((1, -1))
Gr =  nn * np.pi / L * np.cos(nn * np.pi * xx.reshape((-1, 1)) / L)
Grinv = np.linalg.pinv(Gr, rcond = 0)


def B(x, r, h, By_above, Bx_multiplier):
    """ get model magnetic field at arbitrary coordinate """
    x = x + r*0
    r = r + x*0
    x, r = x.flatten().reshape((-1, 1)), r.flatten().reshape((-1, 1))

    r = r - r0

    By = np.zeros_like(x)
    By[r > 0] = np.interp(x[r > 0], xx, By_above) # dBy/dz = 0 above r0

    Br = ( np.exp(-nn * np.pi / L * np.abs(r)) * nn * np.pi / L * np.cos(nn * np.pi * x / L)).dot(h).reshape((-1, 1))
    Bx = (-np.exp(-nn * np.pi / L * np.abs(r)) * nn * np.pi / L * np.sin(nn * np.pi * x / L)).dot(h).reshape((-1, 1))

    Bx[r > 0] *= (-1)

    Br = Br + B0

    return np.hstack([Bx_multiplier*Bx, By, Br])









fig = plt.figure(figsize = (25, 15))
ax3d = fig.add_subplot(131, projection = '3d')
axes = np.array([fig.add_subplot(1, 4, i) for i in [2, 3]])



B0_kwargs = dict(linestyle = '-', linewidth = .5, color = 'grey' , zorder = 0 ) # B0 plot style
B_kwargs  = dict(linestyle = '-', linewidth = 2 , color = 'black', zorder = 50) # B plot style
ionosphereX_kwargs = dict(color = 'orange', linewidth = 4, zorder = 62) # ionosphere line style
ionosphereY_kwargs = dict(color = np.array((166,206,227))/256, linewidth = 4, zorder = 61) # ionosphere line style
flowchannel_kwargs = dict(color = 'olive', linestyle = '-', linewidth = 2, zorder = 63) # ionosphere line style
electron_kwargs = dict(zorder = 100, color = np.array((31,120,180))/256) # electron velocity kwargs
ion_kwargs = dict(zorder = 103, color = np.array((51,160,44))/256, linewidth = 2) # ion velocity kwargs
ground_kwargs = dict(color = 'black', linewidth = 7, zorder  =100) # ground line

xmin, xmax = xx[0], xx[-1]
ymin, ymax = -0.4*r0, 1.8 * r0
ygap = (ymax - ymin) / 10 # whitespace between rows

y_offset_XY    = 1 * (ymax - ymin) + (0  )*ygap 
y_offset_XZ    = 2 * (ymax - ymin) + (2  )*ygap 
y_offset_YZ    = 3 * (ymax - ymin) + (3  )*ygap 
y_offset_lines = 0 * (ymax - ymin) + (0  )*ygap 
line_y0  = y_offset_lines + (ymax - ymin)*.5 - 2*ygap


N_fieldlines = 25 # number of fild lines to plot
N_vectors = 10 # number of electron velocity vectors
indices = [data.time.shape[0] - data.time[data.time > 4.5].shape[0], len(data.time) - 1]
Bx_multipliers = [1e3, 1e3] # apply this multiplier to Bx in the different time steps

maxfac = data['jr'].values[indices[-1]].max()
maxfac = np.round(maxfac * 1e6 * 2) / 2 * 1e-6 # round to nearest .5
maxn   = data['n' ].values[indices[-1]].max() - n0
maxBx  = np.abs(data['Bx_ground' ].values[indices[-1]]).max()
maxBr  = np.abs(data['Br_ground' ].values[indices[-1]]).max()
maxB = np.round(np.max([maxBx, maxBr]) * 1e9 / 10) / 1e8

if True:

    for i in [0, 1]:
        ii = indices[i]

        By = data['By+'].values[ii]
        By_center = By[len(xx)//2]
        h = Grinv.dot(data['Br'].values[ii] - B0) # fourier coefficients
        get_B = lambda t, xyz: B(xyz[0], xyz[2], h, By, Bx_multipliers[i])



        # Plot field lines in XZ plane
        ##############################
        B_lines = []
        B0_lines = []
        for j, x0 in enumerate(np.linspace(xmin, xmax, N_fieldlines)): # trace field lines
            p = np.array([x0, -5*r0]) # start point of the trace

            rk = RK45(get_B, 0, np.array([p[0], 0, p[1]]), t_bound = -r0*1e6, atol = 1e-14, rtol = 1e-9, first_step = 1000, vectorized = True)

            rs = [np.array([p[0], 0, p[1]])]
            while True:
                try:
                    rk.step()
                except:
                    'RK step failed. Breaking out of loop and returning nan'
                    rs.append(rk.y*np.nan)
                    break
                rs.append(rk.y)
                if rs[-1][-1] > r0 or rk.status == 'finished':
                    break

            rs = np.vstack(rs)

            # find the intersection with r0
            s = (r0 - rs[-2][2])/(rs[-1][2] - rs[-2][2])
            rs[-1][0] = s * (rs[-1][0] - rs[-2][0]) + rs[-2][0]
            rs[-1][2] = r0


            B_lines.append(np.vstack((rs[:, 0], rs[:, 2] + y_offset_XZ - ymin)).T)
            B0_lines.append(np.vstack(([x0, x0], np.array([-5*r0, 3*r0]) + y_offset_XZ - ymin)).T)

            # repeat for field lines above the ionosphere:
            p = np.array([rs[-1][0], r0]) # start point of the trace
            rk = RK45(get_B, 0, np.array([p[0], 0, p[1]]), t_bound = -r0*1e6, atol = 1e-14, rtol = 1e-9, first_step = 1000, vectorized = True)


            rs = [np.array([p[0], 0, p[1]])]
            while True:
                try:
                    rk.step()
                except:
                    'RK step failed. Breaking out of loop and returning nan'
                    rs.append(rk.y*np.nan)
                    break
                rs.append(rk.y)
                if rs[-1][-1] > 2*r0 or rk.status == 'finished':
                    break

            rs = np.vstack(rs)

            # find the intersection with r0
            s = (2*r0 - rs[-1][2])/(rs[-2][2] - rs[-1][2])
            rs[-1][0] = s * (rs[-2][0] - rs[-1][0]) + rs[-1][0]
            rs[-1][1] = s * (rs[-2][1] - rs[-1][1]) + rs[-1][1]
            rs[-1][2] = 2*r0

            B_lines.append(np.vstack((rs[:, 0], rs[:, 2] + y_offset_XZ - ymin)).T)


        mask = patches.Rectangle((xmin, y_offset_XZ), (xmax - xmin), (ymax - ymin), facecolor = 'none')
        axes[i].add_patch(mask)

        for l, kws in zip([B_lines, B0_lines], [B_kwargs, B0_kwargs]):
            coll = LineCollection(l, **kws)
            c = axes[i].add_collection(coll)
            c.set_clip_path(mask)



        # Plot field lines in YZ plane
        ##############################
        B_lines = []
        B0_lines = []
        for j, x0 in enumerate(np.linspace(xmin - L//2, xmax + L//2, 2*N_fieldlines - 1)): # trace field lines
            B_lines.append(np.vstack(([x0, x0], np.array([-5*r0, r0]) + y_offset_YZ - ymin)).T)
            B0_lines.append(np.vstack(([x0, x0], np.array([-5*r0, 3*r0]) + y_offset_YZ - ymin)).T)

            B_lines.append(np.vstack(([x0, x0 + Bx_multipliers[i]*By_center/B0*r0], np.array([r0, 2 * r0]) + y_offset_YZ - ymin)).T)



        mask = patches.Rectangle((xmin, y_offset_YZ), (xmax - xmin), (ymax - ymin), facecolor = 'none')
        axes[i].add_patch(mask)

        for l, kws in zip([B_lines, B0_lines], [B_kwargs, B0_kwargs]):
            coll = LineCollection(l, **kws)
            c = axes[i].add_collection(coll)
            c.set_clip_path(mask)



        # Plot ionosphere and ground lines
        ##################################
        for offset, ionosphere_kwargs in zip([y_offset_XZ, y_offset_YZ], [ionosphereX_kwargs, ionosphereY_kwargs]):
            axes[i].plot([xmin, xmax], np.array([r0, r0]) + offset - ymin, **ionosphere_kwargs)
            axes[i].plot([xmin, xmax], np.array([0 ,  0]) + offset - ymin, **ground_kwargs)
            
        # arrowhead and label
        axes[i].scatter(xmax*1.01, y_offset_XZ - ymin + r0, marker=">", color = ionosphereX_kwargs['color'], s = 155, zorder = 80)
        axes[i].scatter(xmax*1.01, y_offset_YZ - ymin + r0, marker=">", color = ionosphereY_kwargs['color'], s = 155, zorder = 80)
        axes[i].text(xmax*1.03, y_offset_XZ - ymin + r0, 'x', color = ionosphereX_kwargs['color'], size = 16, zorder = 80, ha = 'left', va = 'center')
        axes[i].text(xmax*1.03, y_offset_YZ - ymin + r0, 'y', color = ionosphereY_kwargs['color'], size = 16, zorder = 80, ha = 'left', va = 'center')

        # Plot electron and ion velocities
        ##################################
        vex_raw =  data['Ey'].values[ii]/data['Br'].values[ii]
        vey_raw = -data['Ex'].values[ii]/data['Br'].values[ii]

        # electron velocities in XZ and XY planes:
        xs = np.linspace(xmin, xmax, N_vectors)[1:-1]
        vex = np.interp(xs, xx, vex_raw)
        vey = np.full_like(xs, np.interp(xmin + L/2, xx, vey_raw))
        vix = np.interp(xs, xx, data['vix'].values[ii]) 
        viy = np.full_like(xs, np.interp(xmin + L/2, xx, data['viy'].values[ii]))

        axes[i].quiver(xs, y_offset_YZ - ymin + r0, vey, 0, **electron_kwargs, scale = 5e3/L, scale_units = 'x', width = 0.01, clip_on = False)
        axes[i].quiver(xs, y_offset_XZ - ymin + r0, vex, 0, **electron_kwargs, scale = 1e3/L, scale_units = 'x', width = 0.01, clip_on = False)

        # electron and ion velocities in XY plane:
        xs = np.linspace(xmin, xmax, N_vectors*4)[1:-1]
        vex = np.interp(xs, xx, vex_raw)
        vey = np.interp(xs, xx, vey_raw) 
        vix = np.interp(xs, xx, data['viy'].values[ii]) 
        viy = np.interp(xs, xx, data['vix'].values[ii])

        axes[i].quiver(xs, y_offset_XY, vex, vey, **electron_kwargs, scale = v0_max / (ymax - ymin), scale_units = 'y', width = 0.01, clip_on = False)
        axes[i].quiver(xs, y_offset_XY, vex, vey, **electron_kwargs, scale = v0_max / (ymax - ymin), scale_units = 'y', width = 0.01, clip_on = False)
        axes[i].quiver(xs, y_offset_XY, vix, viy, **ion_kwargs,      scale = v0_max / (ymax - ymin), scale_units = 'y', width = 0.01, clip_on = False)
        axes[i].quiver(xs, y_offset_XY, vix, viy, **ion_kwargs,      scale = v0_max / (ymax - ymin), scale_units = 'y', width = 0.01, clip_on = False)

        # x and y axes in XY plane
        axes[i].plot(xx, v0 * (ymax - ymin) / v0_max + y_offset_XY, **flowchannel_kwargs)
        axes[i].plot(xx, np.full_like(xx, y_offset_XY), color = ionosphereX_kwargs['color'], linewidth = 6, zorder = 0)
        axes[i].scatter(xmax, y_offset_XY, marker = '>', color = ionosphereX_kwargs['color'], s = 155, zorder = 103)
        axes[i].text(xmax*1.03, y_offset_XY, 'x', color = ionosphereX_kwargs['color'], size = 16, zorder = 80, ha = 'left', va = 'center')

        axes[i].plot([L//2]*2, [y_offset_XY, (ymax - ymin) + y_offset_XY], **ionosphereY_kwargs)
        axes[i].scatter(xmin + L/2, y_offset_XY + (ymax - ymin)*1.03, marker = '^', color = ionosphereY_kwargs['color'], s = 155, zorder = 100)
        axes[i].text(xmin + L/2*1.03, y_offset_XY + (ymax - ymin)*1.03, 'y', color = ionosphereY_kwargs['color'], size = 16, zorder = 80, ha = 'left', va = 'center')


        # Plot FACs and densities
        #########################
        FACscale = (ymax - ymin) / (maxfac * 2)
        nscale   = (ymax - ymin) / (maxn   * 2)
        Bscale   = (ymax - ymin) / (maxB   * 2) 

        jr = data['jr'].values[ii]
        jrpos = np.maximum(jr, 0)
        jrneg = np.minimum(jr, 0)

        axes[i].plot(xx, np.full_like(xx, line_y0), color = ionosphereX_kwargs['color'], linewidth = 6, zorder = 0)
        axes[i].fill_between(xx, line_y0 + jrpos * FACscale, y2 = line_y0, color = 'red', alpha = .7, linewidth = 0)
        axes[i].fill_between(xx, line_y0 + jrneg * FACscale, y2 = line_y0, color = 'blue' , alpha = .7, linewidth = 0)
        axes[i].plot(xx, line_y0 + data['Bx_ground' ].values[ii] * Bscale, color = 'black', linestyle = '--', label = '$B_x$ ground')
        axes[i].plot(xx, line_y0 + data['Br_ground' ].values[ii] * Bscale, color = 'black', label = '$B_r$ ground')
        axes[i].plot(xx, v0 * (ymax - ymin) / (v0_max * 2) + line_y0, **flowchannel_kwargs)
        axes[i].scatter(xmax, line_y0, marker = '>', color = ionosphereX_kwargs['color'], s = 155, zorder = 103)
        axes[i].text(xmax*1.03, line_y0, 'x', color = ionosphereX_kwargs['color'], size = 16, zorder = 80, ha = 'left', va = 'center')

# Ground mag legend
###################
axes[0].plot([0, L//10], 2*[line_y0 + (ymax - ymin)/2*.9], color = 'black', linestyle = '-')
axes[0].plot([0, L//10], 2*[line_y0 + (ymax - ymin)/2*.7], color = 'black', linestyle = '--')

axes[0].text(L//10, line_y0 + (ymax - ymin)/2 * .9, '$\Delta B_r$ ground', ha = 'left', va = 'center', size = 10, color = 'black')
axes[0].text(L//10, line_y0 + (ymax - ymin)/2 * .7, '$\Delta B_x$ ground', ha = 'left', va = 'center', size = 10, color = 'black')

# FAC annotation
axes[0].annotate('Upward FAC', xy = (L/10*6, line_y0 + (ymax - ymin)/2/2), xycoords = 'data', xytext = (L/10*8, line_y0 + (ymax-ymin)/2*.8), size = 10, va = 'bottom', ha = 'center', zorder = 0, bbox = None, textcoords = 'data', arrowprops=dict(arrowstyle="-", linewidth = 2, connectionstyle="arc3,rad=-.2", color = 'red'))
axes[0].annotate('Downward FAC', xy = (L/10*4, line_y0 - (ymax - ymin)/2/2), xycoords = 'data', xytext = (L/10*2, line_y0 - (ymax-ymin)/2*.8), size = 10, va = 'top', ha = 'center', zorder = 0, bbox = None, textcoords = 'data', arrowprops=dict(arrowstyle="-", linewidth = 2, connectionstyle="arc3,rad=-.2", color = 'blue'))



# Electron ion velocity legend
##############################
axes[0].quiver(L//100, y_offset_XY + (ymax - ymin)*.9, 100, 0, **electron_kwargs, scale = v0_max/ (ymax - ymin), scale_units = 'y', width = 0.01, clip_on = False)
axes[0].quiver(L//100, y_offset_XY + (ymax - ymin)*.8, 100, 0, zorder = 103, color = ion_kwargs['color'], linewidth = 2, scale = v0_max / (ymax - ymin), scale_units = 'y', width = 0.01, clip_on = False)

axes[0].text(L//100, y_offset_XY + (ymax - ymin)*.89, 'electron velocity', ha = 'left', va = 'top', size = 10, color = electron_kwargs['color'])
axes[0].text(L//100, y_offset_XY + (ymax - ymin)*.79, 'ion velocity', ha = 'left', va = 'top', size = 10, color = ion_kwargs['color'])


# Labels for 0 < t < infty column:
##################################

bboxdict = dict(facecolor = 'white', edgecolor = 'white', linewidth = 0, pad = 0, antialiased = True, alpha = .5)
electron_bboxdict = dict(facecolor = 'white', edgecolor = 'white', linewidth = 0, pad = 0, antialiased = True, alpha = .5)
ion_bboxdict = dict(facecolor = 'white', edgecolor = None, linewidth = 0, pad = 0, antialiased = True, alpha = .5)
flowchannel_bboxdict = dict(facecolor = 'white', edgecolor = 'white', linewidth = 0, pad = 0, antialiased = True, alpha = .5)

label0 = u'$\mathbf{B}$ adapts to ionospheric boundary\naccording to Laplace equation'
label1 = u'Electrons partially carry $j_x$,\nchanging $\mathbf{B}$ to make a current $j_y$'
label2 = u'Magnetospere bends magnetic field\n producing a current $j_x$'                                
label3 = u'Feedback: electrons carrying $j_y$\n change the bend'                                      

axes[0].text(L/2, y_offset_XZ - ymin + 0.55*r0, label0, va = 'top', size = 10, ha = 'center', zorder = 200, bbox = bboxdict)

axes[0].annotate(label1, xy=(4*L/9, y_offset_XZ - ymin + r0), xycoords='data', xytext=(L/2, y_offset_XZ - ymin + 1.25*r0), va = 'bottom', size = 10, ha = 'center', zorder = 200, textcoords='data', bbox = electron_bboxdict   , arrowprops=dict(arrowstyle="->", linewidth = 2, connectionstyle="arc3,rad=.2", color = electron_kwargs['color']))
axes[0].annotate(label2, xy=(4*L/9, y_offset_YZ - ymin + r0), xycoords='data', xytext=(L/2, y_offset_YZ - ymin + 1.25*r0), va = 'bottom', size = 10, ha = 'center', zorder = 200, textcoords='data', bbox = flowchannel_bboxdict, arrowprops=dict(arrowstyle="->", linewidth = 2, connectionstyle="arc3,rad=.2", color = flowchannel_kwargs['color']))
axes[0].annotate(label3, xy=(4*L/9, y_offset_YZ - ymin + r0), xycoords='data', xytext=(L/2, y_offset_YZ - ymin +  .75*r0), va = 'top'   , size = 10, ha = 'center', zorder = 200, textcoords='data', bbox = electron_bboxdict   , arrowprops=dict(arrowstyle="->", linewidth = 2, connectionstyle="arc3,rad=.2", color = electron_kwargs['color']))

# connect the boxes:
axes[0].annotate(label2, xy=(  L/8, y_offset_XZ  - ymin + 1.35*r0), xycoords='data', xytext=(L/2, y_offset_YZ - ymin + 1.25*r0), va = 'bottom', size = 10, ha = 'center', zorder = 200, bbox = flowchannel_bboxdict, textcoords = 'data', arrowprops=dict(linewidth = 1, connectionstyle="angle3, angleA=-175, angleB=140", edgecolor = 'black', facecolor = 'grey'))
axes[0].annotate(label1, xy=(8*L/11, y_offset_YZ - ymin + 0.60*r0), xycoords='data', xytext=(L/2, y_offset_XZ - ymin + 1.25*r0), va = 'bottom', size = 10, ha = 'center', zorder = 200, bbox = electron_bboxdict   , textcoords = 'data', arrowprops=dict(linewidth = 1, connectionstyle="angle3, angleA=-175, angleB=140", edgecolor = 'black', facecolor = 'grey'))
axes[0].annotate(label3, xy=(6*L/8, y_offset_YZ  - ymin + 1.45*r0), xycoords='data', xytext=(L/2, y_offset_YZ - ymin +  .75*r0), va = 'top'   , size = 10, ha = 'center', zorder = 200, bbox = electron_bboxdict   , textcoords = 'data', arrowprops=dict(linewidth = 1, connectionstyle="angle3, angleA=-175, angleB=140", edgecolor = 'black', facecolor = 'grey'))

# annotate arrows plot
alabel = u'ions experience\nLorentz force\n$\mathbf{F} = ne(\mathbf{v}_i - \mathbf{v}_e)\\times\mathbf{B}$'
xs = np.linspace(xmin, xmax, N_vectors*4)[1:-1]
vix0 = np.interp(xs, xx, data['vix'].values[indices[0]]) 
viy0 = np.interp(xs, xx, data['viy'].values[indices[0]]) 
kk = 13*len(xs)//27
xscale = L / v0_max/2
yscale = (ymax - ymin) / v0_max
axes[0].annotate(alabel, xy = (xs[kk] + vix0[kk] * xscale, y_offset_XY + viy0[kk] * yscale), xycoords = 'data', xytext = (xs[4*len(xs)//5], y_offset_XY + (ymax - ymin)*.8), size = 10, va = 'bottom', ha = 'center', zorder = 200, bbox = ion_bboxdict, textcoords = 'data', arrowprops=dict(arrowstyle="-", linewidth = 2, connectionstyle="arc3,rad=-.2", color = ion_kwargs['color']))


# Labels for t = infty column:
##############################
label1 = u'$v_{e,x}=0$, $j_x$ is carried by ions'
label3 = u'$v_{e, y}$ matches magnetosphere flow channel'

axes[1].annotate(label1, xy=(4*L/9, y_offset_XZ - ymin + r0), xycoords='data', xytext=(L/2, y_offset_XZ - ymin + 1.25*r0), va = 'bottom', size = 10, ha = 'center', zorder = 200, textcoords='data', bbox = electron_bboxdict   , arrowprops=dict(arrowstyle="->", linewidth = 2, connectionstyle="arc3,rad=.2", color = electron_kwargs['color']))
axes[1].annotate(label3, xy=(4*L/9, y_offset_YZ - ymin + r0), xycoords='data', xytext=(L/2, y_offset_YZ - ymin +  .75*r0), va = 'top'   , size = 10, ha = 'center', zorder = 200, textcoords='data', bbox = electron_bboxdict   , arrowprops=dict(arrowstyle="->", linewidth = 2, connectionstyle="arc3,rad=.2", color = electron_kwargs['color']))

for ax in axes.flatten():
    ax.set_axis_off()
    ax.set_xlim(xmin, xmax*1.03)
    ax.set_ylim(ymin - ygap, (3 + 1)*(ymax - ymin) + 5*ygap)

# set titles for simulation output plots
axes[0].set_title(u'Simulation output at $t = {:.1f}$ s'.format(data.time.values[indices[0]]))
axes[1].set_title(u'Simulation output at $t = \infty$')
if not three_figs:
    axes[0].text(L//2, axes[0].get_ylim()[1], '$\Sigma_P\sim 5, \Sigma_H\sim 5$', va = 'top', ha = 'center')
    axes[1].text(L//2, axes[0].get_ylim()[1], '$\Sigma_P\sim 5, \Sigma_H\sim 5$', va = 'top', ha = 'center')



# Make 3D plot
##############
maxh = 5*r0 # height of flow channel
YFRAC = 10 # size of y axis is 1/YFRAC * size of x axis

xmin3d, xmax3d = xmin - L//2, xmax - L//2
midx = xmin3d + (xmax3d - xmin3d)/2

# y axis
ax3d.plot([midx, midx], [xmin3d/YFRAC, xmax3d/(YFRAC-2)], [r0, r0], **ionosphereY_kwargs)
ax3d.scatter(midx, xmax3d/(YFRAC - 2), r0, marker = (3, 0, 45), color = ionosphereY_kwargs['color'], s = 160, zdir = 'z', zorder = 0)
ax3d.text(midx, xmax3d/(YFRAC - 3), r0, r'y  ($\frac{\partial}{\partial y} = 0$)', color = ionosphereY_kwargs['color'], size = 12, zdir = 'y', zorder = 0)

# x axis
ax3d.plot([xmin3d, xmax3d + L/10], [midx, midx], [r0, r0], **ionosphereX_kwargs)
ax3d.scatter(xmax3d + L/10, midx, r0, marker = (3, 0, 135), color = ionosphereX_kwargs['color'], s = 160, zdir = 'z', zorder = 0)
ax3d.text(xmax3d + L/8, midx, r0, 'x', color = ionosphereX_kwargs['color'], size = 12, zdir = 'x', zorder = 0)



xs = np.linspace(xmin3d, xmax3d, N_fieldlines//4)
for j, x0 in enumerate(xs): # trace field lines
    ax3d.plot([x0, x0], [midx, midx], [-0.1*r0, maxh], linewidth = B_kwargs['linewidth'], color = 'black')

v0_interpolated = np.interp(xs + L//2, xx, v0)
ax3d.scatter(xs, [midx]*len(xs), [r0 + (maxh - r0)/2]*len(xs), marker = 'v', color = B_kwargs['color'], s = 30)
#ax3d.quiver(xs, [midx]*len(xs), [r0 + (maxh - r0)/2]*len(xs), xs*0, v0_interpolated, xs*0, length = 1e3)
ax3d.plot(xx - L//2, v0 / np.max(v0) * 3*r0, np.full_like(xx, maxh), **flowchannel_kwargs)
ax3d.plot([0,  0], [0, 3*r0], [maxh]*2, color = 'grey', linewidth = .5)
ax3d.text(0, 1.5 * r0, maxh, '{:.0f} m/s'.format(data['v0'].max().values), ha = 'center', va = 'center', zdir = 'y', size = 10, zorder = 100)#, bbox = {'facecolor':'white', 'alpha':1, 'pad':0, 'edgecolor':'white'})


# ionosphere and ground surfaces:
X = np.array([xmin3d, xmax3d])
Y = X/YFRAC
X, Y = np.meshgrid(X, Y)
Zi = np.full_like(X, r0)
Zg = np.zeros_like(X)
ax3d.plot_surface(X, Y, Zi, cmap = plt.cm.binary_r, vmin = -.5*r0, vmax = 2*r0, linewidth=0, antialiased = True, zorder = 50, alpha = .3)
ax3d.plot_surface(X, Y, Zg, cmap = plt.cm.binary_r, vmin = -.5*r0, vmax = 2*r0, linewidth=0, antialiased = True, zorder = 50, alpha = .3)

ax3d.text(xmin3d, .8*xmin3d/YFRAC, 0 , 'ground', zdir = 'x' , ha = 'left', va = 'top'   , size = 11, zorder = 100)#, bbox = dict(facecolor = 'white', linewidth = 0, antialiased = True, alpha = .7))
ax3d.text(xmin3d*1.1, .8*xmin3d/YFRAC, r0, r'ionosphere', zdir = 'x', ha = 'left', va = 'top', size = 11, zorder = 100)#, bbox = dict(facecolor = 'white', linewidth = 0, antialiased = True, alpha = .7))
ax3d.text(0, 0, r0 + 1.5*r0, r'$B_0 = {:.0f}$ nT'.format(B0*1e9), zdir = 'z', ha = 'center', va = 'center', size = 11, zorder = 100, rotation = 90)#, bbox = dict(facecolor = 'white', linewidth = 0, antialiased = True, alpha = .7))


ax3d.text(xmin3d + L/4, 0, maxh, 'magnetosphere\nflow channel', zdir = 'y')


ax3d.set_axis_off()


ax3d.get_proj = lambda: np.dot(Axes3D.get_proj(ax3d), np.diag([0.7, 0.7, 1, 1]))
ax3d.view_init(elev = 19, azim = -37, vertical_axis = 'z')
ax3d.dist = 6


# Add scales
############
axes[1].plot([0, L], 2*[y_offset_XZ + 5e3], color = 'grey', linewidth = 2, zorder = 1000)
axes[1].plot([0, 0], y_offset_XZ + np.array([0, 10e3]), color = 'grey', linewidth = 4, zorder = 1000)
axes[1].plot([L, L], y_offset_XZ + np.array([0, 10e3]), color = 'grey', linewidth = 4, zorder = 1000)
axes[1].text(L//2, y_offset_XZ + 5e3, '{:.0f} km'.format(L/1000), bbox = dict(facecolor = 'white', edgecolor = 'white', linewidth = 0, pad = 0, antialiased = True, alpha = 1), size = 10, va = 'center', ha = 'center', zorder = 1001, color = 'grey')

axes[1].plot([-30e3]*2, [y_offset_XZ - ymin, y_offset_XZ - ymin + r0], color = 'grey', linewidth = 2, zorder = 1000, clip_on = False)
axes[1].plot([-20e3, -40e3], [y_offset_XZ - ymin     ]*2, color = 'grey', linewidth = 2, zorder = 1000, clip_on = False)
axes[1].plot([-20e3, -40e3], [y_offset_XZ - ymin + r0]*2, color = 'grey', linewidth = 2, zorder = 1000, clip_on = False)
axes[1].text(-30e3, y_offset_XZ - ymin + r0/2, '{:.0f} km'.format(r0/1000), bbox = dict(facecolor = 'white', edgecolor = 'white', linewidth = 0, pad = 0, antialiased = True, alpha = 1), size = 10, va = 'center', ha = 'center', zorder = 1001, color = 'grey', rotation = 90, clip_on = False)
axes[1].text(20e3, y_offset_XZ - ymin, 'ground', ha = 'left', va = 'center', zorder = 100, size = 11)
axes[1].text(20e3, y_offset_XZ - ymin + r0, 'ionosphere', ha = 'left', va = 'center', zorder = 100, size = 11)

axes[1].plot([300e3]*2, [y_offset_XY, y_offset_XY + ymax - ymin], color = 'grey', linewidth = 2, zorder = 1000, clip_on = False)
axes[1].text(300e3, y_offset_XY + (ymax - ymin) / 2, '{:.0f} m/s'.format(data['v0'].max().values), bbox = dict(facecolor = 'white', edgecolor = 'white', linewidth = 0, pad = 0, antialiased = True, alpha = 1), size = 10, va = 'center', ha = 'center', zorder = 1001, color = 'grey', rotation = 90, clip_on = False)

axes[1].plot([300e3]*2, [line_y0, line_y0 + (ymax - ymin)/2], color = 'grey', linewidth = 2, zorder = 1000, clip_on = False)
axes[1].text(300e3, line_y0 + (ymax - ymin) / 4, '{:.0f} m/s\n{:.1f} $\mu$A/m$^2$\n{:.0f} nT'.format(data['v0'].max().values, maxfac * 1e6, maxB*1e9), bbox = dict(facecolor = 'white', edgecolor = 'white', linewidth = 0, pad = 0, antialiased = True, alpha = 1), size = 10, va = 'center', ha = 'center', zorder = 1001, color = 'grey', rotation = 90, clip_on = False)



# Plot the time series
######################
c1, c2 = 'C0', 'C1'
colors = [c1]*3 + [c2]*3
linestyles = ['--', '-', '-', '--', '-', '-']
linewidths = [1, 5, 1]*2

tsaxes = np.array([plt.subplot2grid((3, 4), (i, 3)) for i in range(3)])

parameters  = ['Bx_ground', 'viy', 'By+']
ylabels = ['$B_x$ on ground [nT]', 'ion velocity in $y$-direction [m/s]', '$B_y$ in space [nT]']
multipliers = [1e9, 1,  1e9]

i = 0
for ax, param, c, label in zip(tsaxes, parameters, multipliers, ylabels):
    for fn, color, ls, lw in zip(fns, colors, linestyles, linewidths):
        data = xr.open_dataset(fn)

        ax.plot(data['time'].values, c * data[param].values[:, data.x.size // 2], linewidth = lw, linestyle = ls, color = color)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid(visible=True, axis = 'x', color = 'grey', linewidth = .5, linestyle = '-')
    ax.set_xlim(0, 120)
    ax.set_ylabel(label, size = 11)

    if three_figs:
        ax.text(-10, ax.get_ylim()[1], 'abc'[i] + ')', clip_on = False, ha = 'right', va = 'top')
    else:
        ax.text(-10, ax.get_ylim()[1], 'd' + str(i+1) + ')', clip_on = False, ha = 'right', va = 'top')
    
    ax.tick_params(axis='both', which='major', labelsize=10)

    i += 1

data = xr.open_dataset(fns[-2])
Sigma_H = np.abs(data['Sigma_H']).values
Sigma_P = np.abs(data['Sigma_P']).values
tsaxes[0].text(data.time.values[-20], multipliers[0] * data[parameters[0]].values[-20, data.x.size//2] + 2, '$\Sigma_H \sim$ {:.0f}'.format(Sigma_H),  color = c2, zorder = 50, ha = 'right', va = 'bottom')
tsaxes[1].text(data.time.values[-20], multipliers[1] * data[parameters[1]].values[-20, data.x.size//2], '$\Sigma_H/\Sigma_P \sim$ {:.1f}'.format(Sigma_H/Sigma_P),  color = 'black', zorder = 50, ha = 'right', va = 'center', size = 10, bbox = {'facecolor':'white', 'linewidth':0, 'pad':0})

data = xr.open_dataset(fns[-1])
Sigma_H = np.abs(data['Sigma_H']).values
Sigma_P = np.abs(data['Sigma_P']).values
tsaxes[1].text(data.time.values[-20], multipliers[1] * data[parameters[1]].values[-20, data.x.size//2], '$\Sigma_H/\Sigma_P \sim$ {:.1f}'.format(Sigma_H/Sigma_P),  color = 'black', zorder = 50, ha = 'right', va = 'center', size = 10, bbox = {'facecolor':'white', 'linewidth':0, 'pad':0})

data = xr.open_dataset(fns[-3])
Sigma_H = np.abs(data['Sigma_H']).values
Sigma_P = np.abs(data['Sigma_P']).values
tsaxes[1].text(data.time.values[-20], multipliers[1] * data[parameters[1]].values[-20, data.x.size//2], '$\Sigma_H/\Sigma_P \sim$ {:.1f}'.format(Sigma_H/Sigma_P),  color = 'black', zorder = 50, ha = 'right', va = 'center', size = 10, bbox = {'facecolor':'white', 'linewidth':0, 'pad':0})


data = xr.open_dataset(fns[1])
Sigma_H = np.abs(data['Sigma_H']).values
Sigma_P = np.abs(data['Sigma_P']).values
tsaxes[0].text(data.time.values[-20], multipliers[0] * data[parameters[0]].values[-20, data.x.size//2] + 2, '$\Sigma_H \sim$ {:.0f}'.format(Sigma_H),  color = c1, zorder = 50, ha = 'right', va = 'bottom')

for fn in fns:
    data = xr.open_dataset(fn)
    Sigma_H = np.abs(data['Sigma_H']).values
    Sigma_P = np.abs(data['Sigma_P']).values
    tsaxes[2].text(data.time.values[-20], multipliers[2] * data[parameters[2]].values[-20, data.x.size//2], '$\Sigma_P \sim$ {:.0f}'.format(Sigma_P),  color = 'black', zorder = 50, ha = 'right', va = 'center', size = 10, bbox = {'facecolor':'white', 'linewidth':0, 'pad':0, 'alpha':.5})


tsaxes[0].set_title('Time series at $x = 0$', pad = 8)
ax.set_xlabel('Time [s]')


if three_figs:
    for i in range(4):
        axes[0].text(-L/20, (4 - i) * (ymax - ymin) + (3 - i) * ygap, 'a' + str(i+1) + ')', clip_on = False, ha = 'right', va = 'top')
        axes[1].text(-L/20, (4 - i) * (ymax - ymin) + (3 - i) * ygap, 'b' + str(i+1) + ')', clip_on = False, ha = 'right', va = 'top')

else:
    # Subplot labels:
    ax3d.set_title('a)', loc = 'left')
    for i in range(4):
        axes[0].text(-L/20, (4 - i) * (ymax - ymin) + (3 - i) * ygap, 'b' + str(i+1) + ')', clip_on = False, ha = 'right', va = 'top')
        axes[1].text(-L/20, (4 - i) * (ymax - ymin) + (3 - i) * ygap, 'c' + str(i+1) + ')', clip_on = False, ha = 'right', va = 'top')


fig.subplots_adjust(top=0.975, bottom=0.06, left=0.005, right=0.985, hspace=0.24, wspace=0.24)


plt.savefig('figures/figure.png', dpi = 250)
plt.savefig('figures/figure.pdf')
plt.show()
