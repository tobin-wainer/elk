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

Path_to_Save_to= '/Users/Tobin/Dropbox/TESS_project/Variability_Statistics/Test_Pipeline_Module/Variability_Metrics/'

Path_to_Read_in_LCs = '/Users/Tobin/Dropbox/TESS_project/Variability_Statistics/Test_Pipeline_Module/Corrected_LCs/'

def Have_Lightcurve(Cluster_name):
    if os.path.exists(Path_to_Read_in_LCs+str(Cluster_name)+'.fits'): 
        return True
    else:
        return False
    
def Read_in_Lightcurve(Cluster_name):
    Corrected_LC=Table.read(Path_to_Read_in_LCs+str(Cluster_name)+'.fits')

    return Corrected_LC

def flux_to_mag(flux):
    m1=10    
    f1=15000    
    mag=2.5*(np.log10(f1/flux)) + m1    
    return mag

def flux_err_to_mag_err(flux, flux_err):
    d_mag_d_flux= -2.5/(flux*np.log(10))

    m_err_squared=(abs(d_mag_d_flux)**2)*(flux_err**2)

    return np.sqrt(m_err_squared)

def get_rms(flux): #Root Mean Squared
    flux_l=list(flux)
    squared_fluxes=[]
    for i in range(len(flux_l)):
        squared_fluxes.append((flux_l[i]**2))
    mean=(np.sum(squared_fluxes)/len(squared_fluxes))
    root=np.sqrt(mean)

    return root

def get_std(flux): #Standard Deviation
    sd=np.std(flux)
    return sd

def get_MAD(flux): #Median Absolute Deviation
    flux_l=list(flux)
    median=np.median(flux_l)
    deviation=abs(flux_l-median)
    sum_deviation=np.sum(deviation)
    med_of_devs= np.median(deviation)

    return med_of_devs

#Generalized to get the range for whatever percentile the heart desires
def get_range(flux, bottom_percentile, upper_percentile):
    flux_l=list(flux)    
    flux_l.sort()    

    bottom_ind=int(len(flux_l)*bottom_percentile)
    upper_ind=int(len(flux_l)*upper_percentile)

    new=flux_l[bottom_ind:upper_ind]
    r= new[-1]-new[0]

    return r  

def skewness(flux):
    sn=scipy.stats.moment(flux, moment=3)
    return sn

def Von_Neumann_Ratio(flux):

    num=[]
    denom=[]
    for i in range(len(flux)-1):
        num.append((flux[i+1] - flux[i])**2/(len(flux)-1))
    for i in range(len(flux)):    
        denom= np.sum((flux[i] - np.mean(flux))**2/(len(flux)-1))

    num_=np.sum(num)
    denom_=np.sum(denom)
    nu= num_/denom_

    return 1/nu

def get_J_Stetson_Stat(light_curve):
    #convert to mag
    mags=light_curve['mag']

    mags_err=light_curve['mag_err']

    delta_times=[]
    for i in range(len(light_curve)-1):
        delta_times.append(light_curve['time'][i+1]-light_curve['time'][i])
    delta_times.append(1)

    light_curve.add_column(Column(delta_times), name=('Delta_Time'), index=4) 
    mean_mag = np.mean(mags)
    n= len(light_curve)

    delta= np.array(sqrt(n/ (n-1)) * ((mags - mean_mag)/mags_err))

    P=[]
    w=[]
    for i in range(len(light_curve)):
        if light_curve['Delta_Time'][i] < .021:
            P.append(delta[i]*delta[i+1])
            w.append(1)
        else:
            P.append(delta[i]**2 -1)
            w.append(0.25)

    sign=[]
    for i in range(len(P)):
        sign.append(math.copysign(1, P[i]))

    individual_J = []
    for i in range(len(P)):
        individual_J.append(w[i]*sign[i]*sqrt(abs(P[i])))

    J= np.sum(individual_J)/np.sum(w)

    return J

def get_LSP(light_curve, Cluster_name, print_figs, save_figs):

    def MonteCarlo_LSP(light_curve):

        def trial(light_curve):

            t = list(light_curve['time'])
            dy = light_curve['flux_err']

            new_f_val=[]
            for i in range(len(light_curve)):
                spread=np.random.normal(0, light_curve['flux_err'][i])
                new_f_val.append(light_curve['flux'][i] + spread)

            omega=np.arange(0.05,11,0.01)
            P_LS = LombScargle(t, new_f_val, dy=dy).power(omega)

            t=Table([omega, P_LS], names=('Frequency', 'Power'))

            return t

        #Here I do a Bootstrap resampling incorporating the flux error to get a confidence interval about the LSP
        def thousand_trials(light_curve):
            list_t=[trial(light_curve) for i in range(1000)]

            power_array=np.zeros((len(list_t), 1095))
            for i in range(len(list_t)):
                for j in range(1095):
                    power_array[i][j]=list_t[i]['Power'][j]

            med=np.median(power_array, axis=0)
            p16=np.percentile(power_array, 16, axis=0)
            p84=np.percentile(power_array, 84, axis=0)

            omega=np.arange(0.05,11,0.01)

            table=Table([omega, p16, med, p84], names=('Frequency', 'Power_16','Power_50', 'Power_84'))

            return table

        mc_table=thousand_trials(light_curve) #Perform a Bootstrap like Monte Carlo experiment to get confidence interal

        mean=np.mean(mc_table['Power_50'])
        two_sigma=2*np.std(mc_table['Power_50'])
        thresh=mean+two_sigma #Arbitrary definition as to what we are calling a significant peak in the LSP   
        peak_ind, pap = find_peaks(mc_table['Power_16'], height=float(thresh)) #Finding the Peaks 

        fig=plt.figure()
        plt.title("LS Periodigram"+str(Cluster_name))
        plt.plot(mc_table['Frequency'], mc_table['Power_16'], linestyle='--', color='k', linewidth=.5)
        plt.plot(mc_table['Frequency'], mc_table['Power_84'], linestyle='--', color='k', linewidth=.5)
        plt.fill_between(mc_table['Frequency'],  mc_table['Power_16'],  mc_table['Power_84'], color='grey', label='$1 \sigma$ Confidence')
        plt.plot(mc_table['Frequency'], mc_table['Power_50'], linestyle='--', color='b', linewidth=1.25, label='Median')
        plt.vlines(x=mc_table['Frequency'][peak_ind], ymin=[0 for i in range(len(pap.get('peak_heights')))], 
                   ymax=pap.get('peak_heights'), linestyle='--', color='g', linewidth=1.25, label='16P > mean+2*sd')
        plt.xscale('log')
        plt.xlabel('Frequency [1/Days]')
        plt.ylabel('Power')
        plt.legend()

        path="Figures/" #Sub Folders in the Path
        which_fig="_LSP"
        out=".png"

        if save_figs:
            plt.savefig(Path_to_Save_to+path+str(Cluster_name)+which_fig+out, format='png') 
        if print_figs:
            plt.show()

        plt.close(fig)

        n_peaks=len(mc_table[peak_ind])

        # 5 Days is around the middle of our time scale, and represents a threshold of how different stars varry (McQuillan et al. 2012)
        # So we want to calculate how much of our power is coming from time scales shorter/longer than 5 days
        average_power_below_5_days=np.mean(mc_table['Power_50'][:495])
        average_power_above_5_days=np.mean(mc_table['Power_50'][495:])

        ratio_of_power_at_high_v_low_freq=round(average_power_above_5_days/average_power_below_5_days,5) 

        # Because each cluster is going to have a different number of peaks, I have an array of len 25, 
        # we will show the number of peaks wach cluster has as a key to how many indices of the array are relevant
        freqs=np.zeros((25))
        pap_a=np.zeros((25))
        for i in range(len(mc_table[peak_ind])):
            freqs[i]=round(mc_table['Frequency'][peak_ind][i],3)
            pap_a[i]=round(pap.get('peak_heights')[i], 4)

        max_LS_power=max(mc_table['Power_50'])
        famp=mc_table['Frequency'][np.where(mc_table['Power_50']==max_LS_power)]

        freqs_with_peaks=freqs[np.where( freqs != 0 )]

        pixel_loc_write=[]
        P_LS_pixel=[]
        omega=np.arange(0.05,11,0.01)

        return fig, (np.array([round(max_LS_power,5), round(famp[0],4), freqs, pap_a, n_peaks, ratio_of_power_at_high_v_low_freq]))

    figure, array= MonteCarlo_LSP(light_curve)

    return figure, array

def autocorr(light_curve, Cluster_name, print_figs, save_figs):

    # Becasue the gap in the middle of the observation can drastically effect the ACF, we only calculate the first have 
    fh_cor_lc=light_curve[:(len(light_curve)//2)]

    acf=statsmodels.tsa.stattools.acf(fh_cor_lc['flux'], nlags=len(fh_cor_lc)-1, alpha=.33)

    err=acf[1]
    ac=acf[0]
    acf_p16=[err[i][0] for i in range(len(err))]
    acf_p84=[err[i][1] for i in range(len(err))]

    deltatimes=[]
    for i in range(len(fh_cor_lc)-1):
        deltatimes.append(fh_cor_lc['time'][i+1]-fh_cor_lc['time'][i])

    plot_times=np.cumsum(deltatimes)

    fig=plt.figure()
    plt.title("ACF"+str(Cluster_name))
    plt.plot(plot_times, ac[:-1])
    plt.fill_between(plot_times, acf_p16[:-1], acf_p84[:-1], color='grey', label='$1 \sigma$ Confidence', alpha=.5)
    plt.xlabel('Delta Time [Days]')
    plt.text(1, .9, str(Cluster_name), fontsize=16)
    path="Figures/"
    which_fig="_ACF_firsthalf"
    out=".png"

    if save_figs:
        plt.savefig(Path_to_Save_to+path+str(Cluster_name)+which_fig+out, format='png')     
    if print_figs:
        plt.show()

    plt.close(fig) 

    first_min_ts=plot_times[argrelextrema(ac, np.less)[0][0]]
    ac_ai=ac[np.where(plot_times > first_min_ts)]
    max_ac=max(ac_ai)
    pa_mac=plot_times[np.where(ac==max_ac)]
    ac_rms= get_rms(ac)

    return fig, (round(max_ac,5), round(pa_mac[0],4), round(ac_rms,5))


def Get_Variable_Stats_Table(Cluster_name, print_figs=True, save_figs=True):   
    #Test to see if I have already downloaded and corrected this cluster, If I have, read in the data
        if Have_Lightcurve(Cluster_name) == True:
            data= Table.read(Path_to_Read_in_LCs+str(Cluster_name)+'.fits')
            
            normalized_flux=np.array(data['flux'])/np.median(data['flux'])

            rms=[np.log10(get_rms(normalized_flux))]
            std=[np.log10(get_std(normalized_flux))]
            MAD=[np.log10(get_MAD(normalized_flux))]
            range_5_95=[np.log10(get_range(normalized_flux, .05, .95))]
            range_1_99=[np.log10(get_range(normalized_flux, .01, .99))]

            j_stat=[get_J_Stetson_Stat(data)]
            sness=[np.log10(abs(skewness(data['flux'])))]
            vnr=[Von_Neumann_Ratio(normalized_flux)]

            max_ac=[]
            period_amac=[]
            ac_rms=[]
            max_LS_power=[]
            freq_amlp=[]
            LS_peak_freq=[]
            LS_peak_power=[]
            LS_n_peaks=[]
            LS_ratio=[]

            ac_fig, ac_arr= autocorr(data, Cluster_name, print_figs, save_figs)
            LS_fig, LS_arr= get_LSP(data, Cluster_name, print_figs, save_figs)

            max_ac.append(ac_arr[0])
            period_amac.append(ac_arr[1])
            ac_rms.append(ac_arr[2])
            max_LS_power.append(LS_arr[0])
            freq_amlp.append(LS_arr[1])
            LS_peak_freq.append(LS_arr[2])
            LS_peak_power.append(LS_arr[3])
            LS_n_peaks.append(LS_arr[4])
            LS_ratio.append(LS_arr[5])

            name___=[Cluster_name]


            Simple_Stats_Table=Table((name___, rms, std, range_5_95, range_1_99, MAD, j_stat, vnr, sness,
                                      max_ac, period_amac, ac_rms, max_LS_power, freq_amlp, LS_peak_freq, LS_peak_power, 
                                      LS_n_peaks, LS_ratio), 
                                      names=('Cluster', 'RMS', 'STD', '5-95_Range', '1-99_Range', 
                                             'MAD', 'Stetson_J_stat', 'Von_Neumann_Ratio', 'Skewness', 'Max_AutoCorr', 'AutoCorr_Period_atM','AutoCorr_rms', 
                                             'Max_LS_Power', 'LS_Freq_atM', 'LS_Peak_Freq','LS_Peak_Power', 'N_LS_Peaks',
                                             'LS_Ratio'))
            
            #Writing out the table
            Simple_Stats_Table.write(Path_to_Save_to+'Stats_Tables/'+str(Cluster_name)+'stats_table.fits', overwrite=True)
  
            return Simple_Stats_Table
        
        else: #This statement means that the lightcuve does not exist in our path
            name___= [Cluster_name]

            std=np.ma.array([0], mask=[1])
            rms=np.ma.array([0], mask=[1])
            range_5_95=np.ma.array([0], mask=[1])
            range_1_99=np.ma.array([0], mask=[1])
            MAD=np.ma.array([0], mask=[1])
            j_stat=np.ma.array([0], mask=[1])
            vnr=np.ma.array([0], mask=[1])
            sness=np.ma.array([0], mask=[1])
            max_ac=np.ma.array([0], mask=[1])
            period_amac=np.ma.array([0], mask=[1])
            ac_rms=np.ma.array([0], mask=[1])
            max_LS_power=np.ma.array([0], mask=[1])
            freq_amlp=np.ma.array([0], mask=[1])
            LS_peak_freq=[np.zeros(25)]
            LS_peak_power=[np.zeros(25)]
            LS_n_peaks=np.ma.array([0], mask=[1])
            LS_ratio=np.ma.array([0], mask=[1])

            Simple_Stats_Table=Table((name___, rms, std, range_5_95, range_1_99, MAD, j_stat, vnr, sness,
                                      max_ac, period_amac, ac_rms, max_LS_power, freq_amlp, LS_peak_freq, LS_peak_power, 
                                      LS_n_peaks, LS_ratio), 
                                      names=('Cluster', 'RMS', 'STD', '5-95_Range', '1-99_Range', 
                                             'MAD', 'Stetson_J_stat', 'Von_Neumann_Ratio', 'Skewness', 'Max_AutoCorr', 'AutoCorr_Period_atM','AutoCorr_rms', 
                                             'Max_LS_Power', 'LS_Freq_atM', 'LS_Peak_Freq','LS_Peak_Power', 'N_LS_Peaks', 
                                             'LS_Ratio')) 
            
            #Writing out the table
            Simple_Stats_Table.write(Path_to_Save_to+'Stats_Tables/'+str(Cluster_name)+'stats_table.fits', overwrite=True) 
       
            return Simple_Stats_Table
            
