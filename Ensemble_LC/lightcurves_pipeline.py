import numpy as np
from astropy.table import Table, join, MaskedColumn, vstack
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy
from astropy.time import Time
import pandas as pd
import re
import seaborn as sns
import datetime
from datetime import datetime
from datetime import timedelta
from math import e
from math import pi
from astropy.table import Column
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import math
from numpy import exp
from scipy import integrate
from scipy.integrate import quad
import pdb
import random
from scipy import stats
from scipy.optimize import curve_fit
import scipy.optimize as opt
import statsmodels 
from multiprocessing import Pool
from scipy.signal import find_peaks
from statsmodels.graphics.tsaplots import plot_acf
import glob
import lightkurve as lk
import numpy as np
import pandas as pd
import imageio
from scipy.signal import argrelextrema

import glob
import lightkurve as lk
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from astropy.timeseries import LombScargle
from astropy.table import Table
import imageio
from lightkurve.correctors import RegressionCorrector
from lightkurve.correctors import DesignMatrix
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import os
import gc
import multiprocessing


Path_to_Save_to= '/Users/Tobin/Dropbox/TESS_project/Variability_Statistics/Test_Pipeline_Module/'

def previously_downloaded(Cluster_name):   
    sub_path= 'Corrected_LCs/' #A sub-folder in my Path
    return os.path.exists(Path_to_Save_to+sub_path+str(Cluster_name)+'output_table.fits')

def tess_data(Callable):
    CLUSTERS = [str(Callable)]

    search = lk.search_tesscut(CLUSTERS[0])#search for the cluster in TESS using lightkurve

    print(CLUSTERS[0]+str(' Has ')+str(len(search))+str(' Observations'))

    if len(search) > 0:
        return True
    else:
        return False 

def degs_to_pixels(degs):
    return degs*60*60/21 #convert degrees to arcsecs and then divide by the resolution of TESS (21 arcsec per pixel)

def pixels_to_degs(pixels):
    return pixels*21/(60*60) #convert degrees to arcsecs and then divide by the resolution of TESS (21 arcsec per pixel)

def flux_to_mag(flux):
    m1=10    
    f1=15000    
    mag=2.5*(np.log10(f1/flux)) + m1    
    return mag

def flux_err_to_mag_err(flux, flux_err):
    d_mag_d_flux= -2.5/(flux*np.log(10))

    m_err_squared=(abs(d_mag_d_flux)**2)*(flux_err**2)

    return np.sqrt(m_err_squared)


def circle_aperture(data,bkg,radius,PERCENTILE):
    radius_in_pixels=degs_to_pixels(radius)
    data_mask = np.zeros_like(data)
    x_len=np.shape(data_mask)[1]
    y_len=np.shape(data_mask)[2]
    #centers
    cen_x=x_len//2
    cen_y=y_len//2
    bkg_mask = np.zeros_like(bkg)
    bkg_cutoff = getUpperLimit(bkg,PERCENTILE)
    for i in range(x_len):
        for j in range(y_len):
            if (i-cen_x)**2+(j-cen_y)**2<(radius_in_pixels)**2:# star mask condition
                data_mask[0,i,j]=1

    x_len=np.shape(bkg_mask)[1]
    y_len=np.shape(bkg_mask)[2]
    cen_x=x_len//2
    cen_y=y_len//2
    for i in range(x_len):
        for j in range(y_len):
            if np.logical_and((i-cen_x)**2+(j-cen_y)**2>(radius_in_pixels)**2, bkg[0,i,j]<bkg_cutoff): # sky mask condition
                bkg_mask[0,i,j]=1            

    star_mask = data_mask==1
    sky_mask = bkg_mask==1
#     sky_mask_2 = bkg_mask_2==1
#     field_mask = field==1
    return star_mask,sky_mask# return masks
    
def getUpperLimit(dataDistribution,PERCENTILE):
    if UPPER_LIMIT_METHOD == 1:
        return np.nanpercentile(dataDistribution, PERCENTILE)

    elif UPPER_LIMIT_METHOD == 2:
        hist = np.histogram(dataDistribution, bins=BINS, range=(0, 3000))# Bin the data
        return hist[1][np.argmax(hist[0])]# Return the flux corresponding to the most populated bin

    elif UPPER_LIMIT_METHOD == 3:
        pass

    elif UPPER_LIMIT_METHOD == 4:
        numMaxima = countMaxima(tpfs[i][frame].flux.reshape((cutout_size, cutout_size)))
        numPixels = np.count_nonzero(~np.isnan(tpfs[i][frame].flux))
        return np.nanpercentile(dataDistribution, 100 - numMaxima / numPixels * 100)

    else:
        return 150
     

UPPER_LIMIT_METHOD = 1
PERCENTILE = 85
cutout_size= 99 #Max for unknown reasons

def downloadable(Callable, current_try_sector):   
    use_name=[Callable]
    #Using a Try statement to see if we can download the cluster data If we cannot
    try:
        tpfs=lk.search_tesscut(use_name[0])[current_try_sector].download(cutout_size=(cutout_size, cutout_size))
    except:
        print("No Download")
        return 'Bad'

    return 'Fine'


def Test_near_edge(tpfs): 

    t1= tpfs[np.where(tpfs.to_lightcurve().quality==0)] #Only selecting time steps that are good, or have quality =0 
    #Also making sure the Sector isn't the one with the Systematic
    if (np.isnan(np.min(t1[0].flux.value)) == False) & (tpfs.sector != 1) & (np.min(t1[0].flux.value) > 1):
        return 'fine'
    else: 
        return 'Bad'


def Test_for_Scattered_Light(use_tpfs, full_model_Normalized):
    # regular grid covering the domain of the data
    X,Y = np.meshgrid(np.arange(0, cutout_size, 1), np.arange(0, cutout_size, 1))
    XX = X.flatten()
    YY = Y.flatten()

    #Define the steps for which we test for scattered light 
    time_steps=np.arange(0,len(use_tpfs), 5) #Set the search to be for every 5th time step, can be changed based of the believed frequency of the scattered light

    coefficients_array=np.zeros((len(time_steps), 3))

    data_flux_values=(use_tpfs-full_model_Normalized).flux.value

    for i in range(len(time_steps)):
        data=data_flux_values[time_steps[i]]
        # best-fit linear plane
        A = np.c_[XX, YY, np.ones(XX.shape)]
        C,_,_,_ = scipy.linalg.lstsq(A, data.flatten())    # coefficients
        coefficients_array[i]=C

        #Deleting defined items we don't need any more to save memory
        del A
        del C
        del data

    X_cos= coefficients_array[:,0]
    Y_cos= coefficients_array[:,1]
    Z_cos= coefficients_array[:,2]

    mxc=max(abs(X_cos))
    myc=max(abs(Y_cos))
    mzc=max(abs(Z_cos))

    #Deleting defined items we don't need any more to save memory
    del X_cos
    del Y_cos
    del Z_cos
    del coefficients_array
    gc.collect() #This is a command which will delete stray arguments to save memory
    
    print(mzc, mxc, myc)

    if (mzc > 2.5) | ((mxc > 0.02) & (myc > 0.02)): 
        return 'Bad'
    else:
        return 'Fine'


def Get_LC(Callable, Radius, Cluster_name):
    # Knowing how many observations we have to work with
    search = lk.search_tesscut(Callable)
    sectors_available=len(search)
    
    #We are also going to doccument how many observations failed each one of our quality tests
    failed_download=0
    near_edge_or_Sector_1=0
    Scattered_Light=0

    #start interating through the observations until I find a good one        
    for current_try_sector in range(sectors_available):
        print("Starting Quality Tests for Observation:", current_try_sector)

        if downloadable(Callable, current_try_sector)== 'Bad':
            print('Failed Download')
            failed_download=failed_download+1
            continue
        else:
            use_name=[Callable]
            tpfs=tpfs=lk.search_tesscut(use_name[0])[current_try_sector].download(cutout_size=(cutout_size, cutout_size))
            
        if Test_near_edge(tpfs) == 'Bad':
            print('Failed Near Edge Test')
            near_edge_or_Sector_1=near_edge_or_Sector_1+1
            continue
        else: 
            use_tpfs = tpfs[np.where(tpfs.to_lightcurve().quality==0)]

        #Getting Rid of where the flux err < 0
        use_tpfs = use_tpfs[use_tpfs.to_lightcurve().flux_err>0]

        #Define the aperature for our Cluster based on previous Vijith functions
        star_mask1=np.empty([len(use_tpfs),cutout_size,cutout_size],dtype='bool')
        sky_mask1=np.empty([len(use_tpfs),cutout_size,cutout_size],dtype='bool')

        star_mask1[0],sky_mask1[0] = circle_aperture(use_tpfs[0].flux.value, use_tpfs[0].flux.value, Radius, PERCENTILE)

        keep_mask=star_mask1[0]
        
        #This will print an image of the cluster with the used aperature shown in red
        p = use_tpfs.plot(frame=use_tpfs.shape[0] // 2, aperture_mask=keep_mask)
        
        #Now we will begin to correct the lightcurve

        # Time average of the pixels in the TPF:
        max_frame = use_tpfs.flux.value.max(axis=0)
        # This renormalizes any columns which are bright because of straps on the detector
        max_frame -= np.median(max_frame, axis=0)
        # This aperture is any "faint" pixels:
        bkg_aper = max_frame < np.percentile(max_frame, PERCENTILE)
        # The average light curve of the faint pixels is a good estimate of the scattered light
        scattered_light = use_tpfs.flux.value[:, bkg_aper].mean(axis=1)

        #We can use our background aperture to create pixel time series and then take Principal Components of the data using Singular Value Decomposition. This gives us the "top" trends that are present in the background data.
        # I have picked 6 based on previous studies showing that is an abritrarily optimal number of components
        pca_dm1 = lk.DesignMatrix(use_tpfs.flux.value[:, bkg_aper], name='PCA').pca(6) 

        #The TESS mission pipeline provides cotrending basis vectors (CBVs) which capture common trends in the dataset. We can use these to detrend out pixel level data.
        #The mission provides MultiScale CBVs, which are at different time scales. In this case, we don't want to use the long scale CBVs, because this may fit out real astrophysical variability. Instead we will use the medium and short time scale CBVs.
        
        cbvs_1 = lk.correctors.cbvcorrector.download_tess_cbvs(sector=use_tpfs.sector, camera=use_tpfs.camera, ccd=use_tpfs.ccd, cbv_type='MultiScale', band=2).interpolate(use_tpfs.to_lightcurve())
        cbvs_2 = lk.correctors.cbvcorrector.download_tess_cbvs(sector=use_tpfs.sector, camera=use_tpfs.camera, ccd=use_tpfs.ccd, cbv_type='MultiScale', band=3).interpolate(use_tpfs.to_lightcurve())


        cbv_dm1 = cbvs_1.to_designmatrix(cbv_indices=np.arange(1, 8))
        cbv_dm2 = cbvs_2.to_designmatrix(cbv_indices=np.arange(1, 8))

        # This combines the different timescale CBVs into a single `designmatrix` object
        cbv_dm_use = lk.DesignMatrixCollection([cbv_dm1, cbv_dm2]).to_designmatrix()
        
        # We can make a simple basis-spline (b-spline) model for astrophysical variability. This will be a flexible, smooth model.
        # The number of knots is important, we want to only correct for very long term variabilty that looks like systematics, so here we have 5 knots, the smaller the better
        spline_dm1 = lk.designmatrix.create_spline_matrix(use_tpfs.time.value, n_knots=5)

        
        # Here we create our design matrix

        dm1 = lk.DesignMatrixCollection([lk.DesignMatrix(scattered_light, name='scattered_light'),
                                         pca_dm1,
                                         cbv_dm_use,
                                         spline_dm1,
                                         ])


        full_model, systematics_model, full_model_Normalized = np.ones((3, *use_tpfs.shape))
        for idx in tqdm(range(use_tpfs.shape[1])):
            for jdx in range(use_tpfs.shape[2]):
                pixel_lightcurve = lk.LightCurve(time=use_tpfs.time.value, flux=use_tpfs.flux.value[:, idx, jdx], flux_err=use_tpfs.flux_err.value[:, idx, jdx])

                #Adding a test to make sure ther are No Flux_err's <= 0
                pixel_lightcurve=pixel_lightcurve[pixel_lightcurve.flux_err > 0]

                r1 = lk.RegressionCorrector(pixel_lightcurve)
                # Correct the pixel light curve by our design matrix
                r1.correct(dm1)
                # Extract just the systematics components
                systematics_model[:, idx, jdx] = (r1.diagnostic_lightcurves['scattered_light'].flux.value +
                                                  r1.diagnostic_lightcurves['CBVs'].flux.value)
                # Add all the components
                full_model[:, idx, jdx] =  (r1.diagnostic_lightcurves['scattered_light'].flux.value +

                                            r1.diagnostic_lightcurves['PCA'].flux.value +
                                            
                                            r1.diagnostic_lightcurves['CBVs'].flux.value +

                                            r1.diagnostic_lightcurves['spline'].flux.value)

                #Making so the model isn't centered around 0
                full_model[:, idx, jdx] -= r1.diagnostic_lightcurves['spline'].flux.value.mean()


                #Making Normalized Model For the Test of Scattered Light
                full_model_Normalized[:, idx, jdx] =  (r1.diagnostic_lightcurves['scattered_light'].flux.value +

                                                       r1.diagnostic_lightcurves['PCA'].flux.value +
                                                       
                                                       r1.diagnostic_lightcurves['CBVs'].flux.value +

                                                       r1.diagnostic_lightcurves['spline'].flux.value)                       

        #Calculate Lightcurves
#NOTE- we are also calculating a lightcurve which does not include the spline model, this is the systematics_model_corrected_lightcurve1
        scattered_light_model_correected_lightcurve=(use_tpfs - scattered_light[:, None, None]).to_lightcurve(aperture_mask=keep_mask)
        systematics_model_corrected_lightcurve=(use_tpfs - systematics_model).to_lightcurve(aperture_mask=keep_mask)
        full_corrected_lightcurve=(use_tpfs - full_model).to_lightcurve(aperture_mask=keep_mask)

        full_corrected_lightcurve_table=Table([full_corrected_lightcurve.time.value, 
                                                full_corrected_lightcurve.flux.value,
                                                full_corrected_lightcurve.flux_err.value], 
                                                names=('time', 'flux', 'flux_err'))
        systematics_model_corrected_lightcurve_table=Table([systematics_model_corrected_lightcurve.time.value, 
                                                systematics_model_corrected_lightcurve.flux.value,
                                                systematics_model_corrected_lightcurve.flux_err.value], 
                                                names=('time', 'flux', 'flux_err'))


        if (Test_for_Scattered_Light(use_tpfs, full_model_Normalized) == 'Bad') & (current_try_sector+1 < sectors_available):
            print("Failed Scattered Light Test")
            Scattered_Light=Scattered_Light+1
            continue
        if (Test_for_Scattered_Light(use_tpfs, full_model_Normalized) == 'Bad') & (current_try_sector+1 == sectors_available):
            print("Failed Scattered Light Test")
            Scattered_Light=Scattered_Light+1
            return ['No Good Observations'], ['Nothing'] , ['Nothing'], np.array(int(sectors_available)), np.array(int(failed_download)), np.array(int(near_edge_or_Sector_1)), np.array(int(Scattered_Light))
        else:
            print(current_try_sector, "Passed Quality Tests")
            #This Else Statement means that the Lightcurve is good and has passed our quality checks
            break
    return full_corrected_lightcurve_table, systematics_model_corrected_lightcurve_table, np.array(int(current_try_sector+1)), np.array(int(sectors_available)), np.array(int(failed_download)), np.array(int(near_edge_or_Sector_1)), np.array(int(Scattered_Light))


def Get_Corrected_Lightcurve(Cluster_name, Location, Radius, Cluster_Age, call_name_type_Name=True, save_figs=True, print_figs=True):
    
    #Setting which collable we want to use
    if call_name_type_Name:
        Callable= Cluster_name
    else:
        Callable= Location
        
    if tess_data(Callable) == True:
        #Test to see if I have already downloaded and corrected this cluster, If I have, read in the data
        if previously_downloaded(Cluster_name) == True:
            output_table= Table.read(Path_to_Save_to+'Corrected_LCs/'+str(Cluster_name)+'output_table.fits')
            if output_table['Used_Ob'][0] != -99: # -99 in the Used Observation Column means there were no good observations
                data= Table.read(Path_to_Save_to+'Corrected_LCs/'+str(Cluster_name)+'.fits')
                systematic_model_data= Table.read(Path_to_Save_to+'Corrected_LCs/'+str(Cluster_name)+'ONLY_SYSMOD.fits')
                
                #Quickly remaking the Light curve plot as an output
                range_=max(data['flux'])-min(data['flux'])
                fig=plt.figure()
                plt.plot(data['time'], data['flux'], color='k', linewidth=.5)
                plt.xlabel('Delta Time [Days]')
                plt.ylabel('Flux [e/s]')
                plt.text(data['time'][0], max(data['flux']), str(Cluster_name), fontsize=16)
                plt.text(data['time'][0], (max(data['flux'])-(range_*.1)),
                         (str("Age:")+str('{0:.2g}').format(Cluster_Age)), fontsize=16)

                if print_figs:
                    plt.show()
                plt.close(fig)
                
                return data, output_table, fig
                      
            else: #When I downloaded it before, there were no good Observations
                data=['No Good Observations'] 
                
                return data, output_table, 'NA'
                              
        
        else: # This else statement refers to the Cluster Not Previously Being Downloaded          
              # So Calling funcion to download and correct data
            data, systematic_model_data, used_observation, Obs_Available, Obs_failed_download, Obs_near_Edge, Obs_Scattered_Light = Get_LC(Callable, Radius, Cluster_name)
        
        # Now that I have my data, if it is a light curve, I'm going to make the figure, and 
        if data[0] != 'No Good Observations':
            data.add_column(Column(flux_to_mag(data['flux'])), name='mag')
            data.add_column(Column(flux_err_to_mag_err(data['flux'], data['flux_err'])), name='mag_err')

            #Making the Output Table
            name___=[Cluster_name]
            HTD=[True]
            OB_use=[used_observation]
            OB_av=[Obs_Available]
            OB_fd=[Obs_failed_download]
            OB_ne=[Obs_near_Edge]
            OB_sl=[Obs_Scattered_Light]


            output_table=Table([name___, [Location], [Radius], [Cluster_Age], HTD, OB_use, OB_av, OB_fd, OB_ne, OB_sl],
                               names=('Name', 'Location', 'Radius [deg]', 'Log Age', 'Has_TESS_Data', 'Used_Ob',
                                      'Obs_Available', 'Obs_DL_Failed', 'Obs_Near_Edge_S1', 'Obs_Scattered_Light'))

            #Writting out the data, so I never have to Download and Correct again, but only if there is data
            lc_path="Corrected_LCs/"        
            data.write(Path_to_Save_to+lc_path+str(Cluster_name)+'.fits')
            systematic_model_data.write(Path_to_Save_to+lc_path+str(Cluster_name)+'ONLY_SYSMOD.fits')
            output_table.write(Path_to_Save_to+lc_path+str(Cluster_name)+'output_table.fits')

            #Now I am going to save a plot of the light curve to go visually inspect later
            range_=max(data['flux'])-min(data['flux'])
            fig=plt.figure()
            plt.plot(data['time'], data['flux'], color='k', linewidth=.5)
            plt.xlabel('Delta Time [Days]')
            plt.ylabel('Flux [e/s]')
            plt.text(data['time'][0], max(data['flux']), str(Cluster_name), fontsize=16)
            plt.text(data['time'][0], (max(data['flux'])-(range_*.1)),
                     (str("Age:")+str('{0:.2g}').format(Cluster_Age)), fontsize=16)

            path="Figures/LCs/" #Sub-folder 
            which_fig="Full_Corrected_LC"
            out=".png"

            if save_figs:
                plt.savefig(Path_to_Save_to+path+str(Cluster_name)+which_fig+out, format='png') 
            if print_figs:
                plt.show()
            plt.close(fig) 

            range_=max(systematic_model_data['flux'])-min(systematic_model_data['flux'])
            fig2=plt.figure()
            plt.plot(systematic_model_data['time'], systematic_model_data['flux'], color='k', linewidth=.5)
            plt.xlabel('Delta Time [Days]')
            plt.ylabel('Flux [e/s]')
            plt.text(systematic_model_data['time'][0], max(systematic_model_data['flux']), str(Cluster_name), fontsize=16)
            plt.text(systematic_model_data['time'][0], (max(systematic_model_data['flux'])-(range_*.1)),
                     (str("Age:")+str('{0:.2g}').format(Cluster_Age)), fontsize=16)

            which_fig2="Only_Systematic_Model_Corrected_LC"

            if save_figs:
                plt.savefig(Path_to_Save_to+path+str(Cluster_name)+which_fig2+out, format='png') 
            plt.close(fig2) 
            
            return data, output_table, fig
                
            
        else: #This else statement refers to there being No good Observtions

            used_observation= -99

            #Making the Output Table
            name___=[Cluster_name]
            HTD=[True]
            OB_use=[used_observation]
            OB_av=[Obs_Available]
            OB_fd=[Obs_failed_download]
            OB_ne=[Obs_near_Edge]
            OB_sl=[Obs_Scattered_Light]


            output_table=Table([name___, [Location], [Radius], [Cluster_Age], HTD, OB_use, OB_av, OB_fd, OB_ne, OB_sl],
                               names=('Name', 'Location', 'Radius [deg]', 'Log Age', 'Has_TESS_Data', 'Used_Ob',
                                      'Obs_Available', 'Obs_DL_Failed', 'Obs_Near_Edge_S1', 'Obs_Scattered_Light'))
            
            output_table.write(Path_to_Save_to+"Corrected_LCs/"+str(Cluster_name)+'output_table.fits')
            
            return data, output_table, 'NA'
          
    else: #This Means that there is no TESS coverage for the Cluster (Easiest to check)
        name___= [Cluster_name]       
        HTD=[False]  
        OB_use= np.ma.array([0], mask=[1])
        OB_av= np.ma.array([0], mask=[1])
        OB_fd= np.ma.array([0], mask=[1])
        OB_ne= np.ma.array([0], mask=[1])
        OB_sl= np.ma.array([0], mask=[1])        

        output_table=Table([name___, [Location], [Radius], [Cluster_Age], HTD, OB_use, OB_av, OB_fd, OB_ne, OB_sl],
                           names=('Name', 'Location', 'Radius [deg]', 'Log Age', 'Has_TESS_Data', 'Used_Ob',
                                  'Obs_Available', 'Obs_DL_Failed', 'Obs_Near_Edge_S1', 'Obs_Scattered_Light'))

        output_table.write(Path_to_Save_to+"Corrected_LCs/"+str(Cluster_name)+'output_table.fits', Overwrite=True)

        return 'No TESS DATA', output_table, 'NA'
        
        
