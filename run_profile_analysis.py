import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import os

import fns_binning_demo as fns_binning

N_bins_r = 30

mode_plot_wedges = True

N_stacks_plot = 5
resolution = 4
um_per_pixel = 1/1.6028
radius_plot_wedges = 200

directory = 'data_demo'
markers = ['Sox2','Lmx1a']

N_marker = len(markers)
colormaps = ['binary','Greens']

timepoints = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

print('STATUS: analysing ' + directory + '\n with markers ' + str(markers) + '\n and timepoints ' + str(timepoints) )

for tpoint in timepoints:
    
    print('--- timepoint ' + tpoint)
    
    subdirectory_data_tpoint = directory + '/' + tpoint

    subdirectory_data_tpoint_check = subdirectory_data_tpoint + '/' + markers[1]
    stacks = [name for name in os.listdir(subdirectory_data_tpoint_check) if os.path.isfile(os.path.join(subdirectory_data_tpoint_check, name))]
    N_stacks = len(stacks)

    plt.close('all')
    H,W = 2*N_marker,N_stacks_plot
    fig_size = [15,8]     
    fs = 10
    params = {
              'figure.figsize': fig_size,
              'font.size':   fs,
              'xtick.labelsize': fs,
              'ytick.labelsize': fs,
              }
    plt.rcParams.update(params)
    plt.rcParams['pcolor.shading'] = 'nearest' #needed for pcolor to avoid errors

    profiles_all = np.zeros((N_stacks,N_bins_r,N_marker))
    
    plt.figure()
    for it in range(0,N_stacks_plot):
        
        print('stack' + str(it))
        
        for it_marker in range(0,N_marker):
            
            marker = markers[it_marker]
            subdirectory_data_tpoint_marker = subdirectory_data_tpoint + '/' + markers[it_marker]
 
            field = np.loadtxt(subdirectory_data_tpoint_marker + '/' + stacks[it])
            field = field[::resolution,::resolution] #such that [i,j] are x,y entries
            
            N_x,N_y = field.shape
            max_val = 200
            
            x = np.linspace(0,N_x-1,N_x)*resolution*um_per_pixel
            y = np.linspace(0,N_y-1,N_y)*resolution*um_per_pixel
            
            x_min = min(x)
            x_max = max(x)

            if it_marker == 0: #calculate center based on Sox2
                
                center = fns_binning.calc_center(x,y,field)

            bins_r,intensity_radius = fns_binning.calc_profile(x,y,field,center,N_bins_r)

            profiles_all[it,:,it_marker] = intensity_radius

            if it < N_stacks_plot:
                ax=plt.subplot(H,W,it+it_marker*N_stacks_plot+1)
                plt.imshow(np.rot90(field,k=1), extent=[x_min,x_max,x_min,x_max], cmap='binary',vmin=0,vmax=max_val)
                
                plt.plot(center[0],center[1],'x',color='k')
                
                plt.xticks([])
                plt.yticks([])

                plt.subplot(H,W,it+(it_marker+N_marker)*N_stacks_plot+1)

                plt.plot(bins_r,intensity_radius)

                plt.yticks([])
                plt.xlabel(r'radius ($\mu$m)')
                plt.ylabel(marker)

        plt.tight_layout()
