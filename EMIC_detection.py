# AUTHOR: Taylor Whitney Aegerter
# PURPOSE: Identifying EMIC waves using PySPEDAS to pull and save data
#   Based on Bortnik et al. (2007)
#   Uses Pedersen et al. (2024) geomagnetic storm list
# Current as of: 24 March 2026

import pyspedas
import pytplot
import numpy as np
from scipy import signal, interpolate
import matplotlib.pyplot as plt
import csv

file_path = 'C:/Users/tawh2779/OneDrive/Documents/CU/Research/EMIC_All Storms/Initial Phase Study/Adjusted Phase Times' 

Re = 6371. #km, Earth's radius

# %% Function for calculating gyrofrequencies

def fcp(mag_data, avg_fs):
    
    qp = 1.602e-19 #coulombs, proton charge
    mp = 1.673e-27 #kg, proton mass
    
    fcp_arr = (qp * mag_data * 1e-9)/(mp * 2 * np.pi)
    
    fcp_arr_avg = []
    
    for i in range(0, len(mag_data), avg_fs):
        fcp_arr_avg.append(np.average(fcp_arr[i:i+avg_fs]))
    
    return np.array(fcp_arr_avg, dtype='float64')

# %% Function for calculating spectrograms
    #spectrogram is a legacy function; if it stops working, there is also scipy.signal.ShortTimeFFT.spectrogram

def spec(mag_data, fs, window):
    freq, t, spec = signal.spectrogram(mag_data[1], fs=fs, window='hann', nperseg=window)
    spec = np.transpose(spec)
    
    t_adj = []
    for i in range(len(t)):
        t_adj.append(mag_data[0][int(fs*t[i])])
    
    return freq, np.array(t_adj), spec

# %% Calculating average magnetopause location from Shue et al. (1998)

def avg_magpause(Bz_arr, Dp_arr, t_arr):
    
    trange_ind = np.where((t_arr >= trange[0]) & (t_arr <= trange[1]))[0]

    zenith = np.linspace(-np.pi, np.pi, num = 100)
    r0 = np.zeros(len(trange_ind)) * np.nan
    alpha = np.zeros(len(trange_ind)) * np.nan
    r = np.zeros([len(trange_ind), 100]) * np.nan

    np.seterr(divide='ignore', invalid='ignore')

    for t in range(len(trange_ind)):
        r0[t] = (10.22 + 1.29*np.tanh(0.184*(Bz_arr[trange_ind[t]] + 8.14))) * Dp_arr[trange_ind[t]]**(-1/6.6)
        alpha[t] = (0.58 - 0.007*Bz_arr[trange_ind[t]]) * (1 + 0.024*np.log(Dp_arr[trange_ind[t]]))

        r[t,:] = r0[t] * (2 / (1 + np.cos(zenith))) ** alpha[t]

    r_avg = np.nanmean(r, axis=0)
    # r_std = np.nanstd(r, axis=0)
    r_min = np.nanmin(r, axis=0)
    r_max = np.nanmax(r, axis=0)

    x_avg = r_avg * np.cos(zenith)
    y_avg = r_avg * np.sin(zenith)
    
    x_min = r_min * np.cos(zenith)
    y_min = r_min * np.sin(zenith)
    
    x_max = r_max * np.cos(zenith)
    y_max = r_max * np.sin(zenith)

    return x_avg, y_avg, x_min, y_min, x_max, y_max, r_avg, r_min, r_max


# %% Converting data from different sized time arrays to align with a specified time array

def time_align(og_t, align_t, data):
    # og_t is the time array of the data array
    # align_t is the time array that the data will be aligned to
    
    n_t = len(align_t)
    if data.ndim > 1: 
        n_data = data.shape[1]
        new_data = np.zeros((n_t, n_data)) * np.nan
    else: 
        n_data = 1
        new_data = np.zeros(n_t) * np.nan
    
    if len(og_t) > len(align_t): # if the data array is a larger size than the time array it is being aigned to, find the nearest time values
        new_t = np.zeros(n_t)
        for t in range(n_t):
            new_t_ind = (np.abs(og_t - align_t[t])).argmin()
            new_t[t] = og_t[new_t_ind]
            if n_data > 1: new_data[t,:] = data[new_t_ind]
            else: new_data[t] = data[new_t_ind]
    else: # if the data array is a smaller size than the time array it is being aligned to, interpolate the data to fit the same size
        new_t_ind = np.where((align_t >= og_t[0]) & (align_t <= og_t[-1]))[0]
        if n_data > 1: 
            for d in range(n_data):
                interp = interpolate.interp1d(og_t, data[:,d-1])
                new_data[new_t_ind,d-1] = interp(align_t[new_t_ind])
        else: 
            interp = interpolate.interp1d(og_t, data)
            new_data[new_t_ind] = interp(align_t[new_t_ind])
        new_t = np.zeros(n_t) * np.nan
        new_t[new_t_ind] = align_t[new_t_ind]
    
    new_t[np.isnan(new_t)] = align_t[np.isnan(new_t)] #if the align_t data has time on either end beyond the og_t data, it will keep them as NaNs; this reassigns those time values as the align_t value (the data will still be NaN)
    
    return [new_t, new_data]

# %% Identify magnetopause crossings based on model-derived magnetopause

def magpause(r_min, x_coord, y_coord, imag_arr, l_shell, l_min):
    # imag_arr should be the array of 1s and 0s from the data indicating when the spacecraft is within the magnetosphere
    # r_avg and r_std are of size 100, ranging from calculations with 0 to 2pi
    
    r_sat = np.sqrt(x_coord**2 + y_coord**2)
    ssp_sat = np.arctan2(y_coord, x_coord) # spacecraft angle from subsolar point
    zenith = np.linspace(-np.pi, np.pi, num=100) # to match the solar zenith used for the magnetopause calculations

    for r in range(len(imag_arr)):
        ang_ind = (np.abs(ssp_sat[r] - zenith)).argmin()
        r_inner = r_min[ang_ind] # inner boundary of the magnetopause range
        if r_sat[r] > r_inner or l_shell[r] < l_min: imag_arr[r] = 0 # spacecraft is outside the magnetopause

    return imag_arr

# %% Wave identification algorithm (Bortnik et al 2007); Background (median) extraction

def median_extraction(t_arr, freq_arr, spec_arr):
    
    trange_ind = np.where((t_arr >= trange[0]) & (t_arr <= trange[1]))[0]
    
    median = []
    std = []
    for f in freq_arr:
        f_ind = np.where(freq_arr == f)[0]
        median.append(np.nanmedian(spec_arr[trange_ind, f_ind]))
        std.append(np.nanstd(spec_arr[trange_ind, f_ind]))
    
    return np.array(median), np.array(std)

# %% Create a mask for times with broadband activity; based on the shape of the spectrogram frequency profile at a given time (PSD)

def broadband_mask(spec_arr, freq_arr, t_arr, fcp_arr, msn):
    # spec_arr should be the array with the median subtracted

    tind = np.where((t_arr >= trange[0]) & (t_arr <= trange[1]))[0]
    broadband = np.zeros(len(tind), dtype=int)
    thresh = 1e1 #threshold, requires detected signals to be at least one order of magnitude greater than the background
    
    mask = [np.nan]
    
    for t in range(len(tind)):
        
        if np.all(np.isnan(spec_arr[tind[t]])): continue

        f_cutoff = np.where((freq_arr >= fcp_arr[tind[t]]/16) & (freq_arr <= fcp_arr[tind[t]]))[0] #gyrofrequency range; between the Oxygen and Hydrogen gyrofrequencies
        t_cutoff = np.where((t_arr >= t_arr[tind[t]]-5*60) & (t_arr <= t_arr[tind[t]]))[0] #time range; previous 5 minutes of data
        
        if len(f_cutoff) == 0 or len(t_cutoff) == 0: continue
        
        #for all but GOES, mask if: the whole normalized PSD is above the threshold, OR more than 75% of the PSD above the Hydrogen gyrofrequency is above the threshold
        if msn != 'goes':
            if np.all(spec_arr[tind[t]] >= thresh): broadband[t] = 1
            if len(np.where(spec_arr[tind[t]][f_cutoff[-1]:] >= thresh)[0]) > 0.75*len(spec_arr[tind[t]][f_cutoff[-1]:]):
                broadband[t] = 1
        #for all missions, mask if: signal below the Oxygen gyrofrequency is higher than the max in the gyrofrequency range, less than 15% of the data in the gyrofrequency range is above the threshold, AND the gyrofrequency range is less than 50% of the entire frequency range
        if np.any(spec_arr[tind[t]][:f_cutoff[0]] >= np.nanmax(spec_arr[tind[t]][f_cutoff])) and len(np.where(spec_arr[tind[t]][f_cutoff] >= thresh)[0]) <= 0.15*len(f_cutoff) and len(f_cutoff) <= 0.5*len(freq_arr): 
            broadband[t] = 1
        #for MMS, mask if: the portion of the frequency range above the threshold is more than 2.5x the length of the gyrofrequency range, OR the gyrofrequency range is less than 75% of the frequency range AND
        # more than 75% of the signal is above the threshold, anything above the gyrofrequency range is above the threshold and the maximum within the gyrofrequency range, OR no more than 5 data points in the top 25% of the frequency range are above the threshold
        if msn == 'mms':
            if len(np.where(spec_arr[tind[t]] >= thresh)[0]) >= 2.5*len(f_cutoff):
                broadband[t] = 1
            if fcp_arr[tind[t]] < 0.75*freq_arr[-1]:
                if len(np.where(spec_arr[tind[t]] >= thresh)[0]) > 0.75*len(spec_arr[tind[t]]): 
                    broadband[t] = 1
                if np.any(spec_arr[tind[t]][f_cutoff[-1]:] > thresh) and np.any(spec_arr[tind[t]][f_cutoff[-1]:] > np.nanmax(spec_arr[tind[t]][f_cutoff])):
                    broadband[t] = 1
                if len(np.where(spec_arr[tind[t]][np.where(freq_arr > 0.75*freq_arr[-1])[0]] >= thresh)[0]) >= 5:
                    broadband[t] = 1
        if len(np.where(broadband[t_cutoff-tind[0]] == 1)[0]) >= 0.4*len(t_cutoff): #40% or more of the data within the last 5 minutes is broadband
            if msn == 'mms': 
                mms_t_cutoff = np.where((t_arr >= t_arr[tind[t]]-6*60) & (t_arr <= t_arr[tind[t]]+1*60))[0]
                mask.extend(t_arr[mms_t_cutoff])
            else: mask.extend(t_arr[t_cutoff])     
            
    mask = sorted(list(set(mask[1:])))
    
    return mask

# %% Wave identification algorithm (Bortnik et al 2007); Sliding average/spectral peak identification

def sliding_avg(spec_arr, freq_arr, fs, f_low, f_high, max_f):
    # spec_arr should be the array with the median subtracted
    
    w = 0.01*fs #sliding-average window, 1% of the sampling frequency
    thresh = 1e1 #threshold, requires detected signals to be at least one order of magnitude greater than the background
    
    cutoff = np.where((freq_arr >= 1*f_low) & (freq_arr <= 1*f_high))[0]
    f = freq_arr[cutoff] #frequencies within the upper/lower bounds
    
    min_width = 0.125 #Hz, minimum width of the spectral peak
    max_width = (f_high - f_low) * 0.8 #Hz, maximum width of spectral peak

    avg = np.empty(cutoff[0]) * np.nan
    std = np.empty(cutoff[0]) * np.nan
    
    for i in range(0, len(spec_arr[cutoff])):
        ind = np.where((f >= (f[i]-0.5*w)) & (f <= (f[i]+0.5*w)))
        avg = np.append(avg, np.nanmean(spec_arr[cutoff][ind]))
        std = np.append(std, np.nanstd(spec_arr[cutoff][ind]))
    
    peak = f_bot = f_top = f_max = -9999
    
    if np.any(avg >= thresh) and np.any(spec_arr < thresh):
        if np.all(spec_arr >= thresh): 
            peak = f_bot = f_top = f_max = -9999
        elif len(np.where(spec_arr >= thresh)[0]) >= 2.5*len(cutoff):
            peak = f_bot = f_top = f_max = -9999
        elif np.any(spec_arr[cutoff[-1]:] > thresh) and np.any(spec_arr[cutoff[-1]:] > np.nanmax(avg)):
            peak = f_bot = f_top = f_max = -9999
        else: peak = np.where(avg >= thresh)[0]
        
        # accounting for multiple peaks above the threshold; uses only the highest peak
        if type(peak) != int and np.any(np.diff(peak) > 1):
            dip = np.where(np.diff(peak) > 1)[0]
            if np.any(np.diff(peak) > 1):
                dip = np.where(np.diff(peak) > 1)[0] +1
                peak1 = peak[:dip[0]]
                if len(dip) == 1: 
                    peak2 = peak[dip[0]:]
                    if np.max(avg[peak1]) > np.max(avg[peak2]): peak = peak1
                    elif np.max(avg[peak2]) > np.max(avg[peak1]): peak = peak2
                elif len(dip) == 2:
                    peak2 = peak[dip[0]:dip[1]]
                    peak3 = peak[dip[1]:]
                    if np.max(avg[peak1]) > np.max(avg[peak2]) and np.max(avg[peak1]) > np.max(avg[peak3]): peak = peak1
                    elif np.max(avg[peak2]) > np.max(avg[peak1]) and np.max(avg[peak2]) > np.max(avg[peak3]): peak = peak2
                    elif np.max(avg[peak3]) > np.max(avg[peak1]) and np.max(avg[peak3]) > np.max(avg[peak2]): peak = peak3
                elif len(dip) == 3:
                    peak2 = peak[dip[0]:dip[1]]
                    peak3 = peak[dip[1]:dip[2]]
                    peak4 = peak[dip[2]:]
                    if np.max(avg[peak1]) > np.max(avg[peak2]) and np.max(avg[peak1]) > np.max(avg[peak3]) and np.max(avg[peak1]) > np.max(avg[peak4]): peak = peak1
                    elif np.max(avg[peak2]) > np.max(avg[peak1]) and np.max(avg[peak2]) > np.max(avg[peak3]) and np.max(avg[peak2]) > np.max(avg[peak4]): peak = peak2
                    elif np.max(avg[peak3]) > np.max(avg[peak1]) and np.max(avg[peak3]) > np.max(avg[peak2]) and np.max(avg[peak3]) > np.max(avg[peak4]): peak = peak3
                    elif np.max(avg[peak4]) > np.max(avg[peak1]) and np.max(avg[peak4]) > np.max(avg[peak2]) and np.max(avg[peak4]) > np.max(avg[peak3]): peak = peak4
                elif len(dip) > 3: peak = f_bot = f_top = f_max = -9999
        
        if type(peak) != int and freq_arr[peak[-1]] > freq_arr[peak[0]]+min_width and freq_arr[peak[-1]] < freq_arr[peak[0]]+max_width:
            f_bot = peak[0]
            f_top = peak[-1]
            f_max = np.where(avg == np.nanmax(avg))[0][0]
            if freq_arr[f_max] > max_f: f_bot = f_top = f_max = -9999

    return f_bot, f_max, f_top, avg, std
    
# %% Wave identification algorithm (Bortnik et al 2007); Finding spectral peaks for entire phase

def spec_peaks(t_arr, spec_arr, freq_arr, fs, f_low, f_high, max_f, mask_t):#, l_shell, l_min, imag):
    # spec_arr should be the array with the median subtracted
    # mask should be 2d array of both start and end times

    tind = np.where((t_arr >= trange[0]) & (t_arr <= trange[1]))[0]
    
    f_bot = np.zeros(len(tind), dtype=int) #lower threshold of the spectral peak
    f_max = np.zeros(len(tind), dtype=int) #maximum frequency of the spectral peak
    f_top = np.zeros(len(tind), dtype=int) #upper threshold of the spectral peak
    peaks_all = np.zeros([3, len(tind)]) * np.nan

    peaks_ind = []
    peaks_ind.append(-9999) #gives the array an initial value so that the indexing works out; will be trimmed

    t_sep = 2*60 #2 minute separation between each wave
    blk_ind = 1

    mask_ind = []
    for m in range(len(mask_t)): mask_ind.append(np.where(t_arr == mask_t[m])[0][0])
    broadband_median, broadband_std = median_extraction(np.array(mask_t), freq_arr, spec_arr[mask_ind])
    spec_arr[mask_ind,:] = spec_arr[mask_ind,:]/broadband_median

    for t in range(len(tind)):
        cutoff = np.where((freq_arr >= 1*f_low[tind[t]]) & (freq_arr <= 1*f_high[tind[t]]))[0]
        if len(cutoff) == 0: continue
        f_bot[t], f_max[t], f_top[t], avg, std = sliding_avg(spec_arr[tind[t],:], freq_arr, fs, f_low[tind[t]], f_high[tind[t]], max_f)
    
        if f_max[t] != -9999: 
            peaks_all[0,t] = t_arr[tind[t]] #time of the peak
            peaks_all[1,t] = freq_arr[f_max[t]] #frequency of the peak
            peaks_ind.append(tind[t])
            if peaks_ind[-1] != -9999 and peaks_ind[-2] != -9999 and \
            t_arr[peaks_ind[-1]] - t_arr[peaks_ind[-2]] > t_sep:
                blk_ind += 1
            peaks_all[2,t] = blk_ind #block index for wave association across times
        else: peaks_all[:,t] = np.nan


    if np.isnan(peaks_all[2]).all(): n_peaks_all = 0
    else: n_peaks_all = int(np.nanmax(peaks_all[2]))
    peaks_times = np.zeros([2, n_peaks_all]) * np.nan
    
    peaks = peaks_all.copy()
    
    for j in range(n_peaks_all):
        times = np.where(peaks_all[2] == j+1)[0]

        if np.all(np.diff(peaks_all[1,times]) == 0):
            # removing instrumentation effects in the data (if all the peaks are exactly the same frequency, assume that the data has an unphysical signature)
            peaks[:,times] = np.nan
            peaks_times[:,j] = np.nan
            continue

        for k in range(len(times)):
            # testing association 
            if k+1 == len(times) or len(times) == 0: continue
            if freq_arr[f_bot[times[k+1]]] < freq_arr[f_top[times[k]]] and \
                freq_arr[f_bot[times[k]]] < freq_arr[f_top[times[k+1]]]:
                continue
            else: 
                peaks[:,times[k]] = np.nan
                
        times = np.where(peaks[2] == j+1)[0]
        if len(times) == 0: continue
        peaks_times[0,j] = peaks[0,times[0]] #start times
        peaks_times[1,j] = peaks[0,times[-1]] #end times
        
         
        if peaks_times[1,j] - peaks_times[0,j] <= 1.5*60 or len(times) <= 2:
            # removing impulsive bursts (requires each wave to be at least 1.5 minutes long or have more than 2 data points)
            peaks[:,times] = np.nan
            peaks_times[:,j] = np.nan
        
    n_peaks = np.count_nonzero(peaks_times[0][~np.isnan(peaks_times[0])]) #also need to remove nan values from peaks_times and peaks
    if n_peaks == 0: 
        peaks = np.array([[0],[0],[0]])
        peaks_times = np.array([[0],[0]])
    
    return peaks_ind[1:], peaks[:, ~np.isnan(peaks).any(axis=0)], peaks_times[:, ~np.isnan(peaks_times).any(axis=0)], n_peaks#, peaks_all

# %% Importing Pederson et al. (2024) list of geomagnetic storms

list_path = 'C:/Users/tawh2779/OneDrive/Documents/CU/Research/Background/Pederson_list.txt' #LASP computer

storm_id = []
storm_id.append(-9999)
initial_phase = []
main_phase = []
min_symh = []
end_time = []
symh_min_val = []

with open(list_path, 'r') as data:
    csv_reader = csv.reader(data, delimiter=' ')
    for i in range(20): #number of header lines
        data.readline() #skip header lines
    for column in csv_reader:
        if any(column) == False: continue #skip blank lines
        if column[3] != '' and int(column[3]) < 10: col_ind = 3
        elif column[2] != '' and int(column[2]) < 100: col_ind = 2
        else: col_ind = 1
        if column[col_ind] != '' and int(column[col_ind]) != storm_id[-1]:
            storm_id.append(int(column[col_ind]))
            if column[col_ind+5][0] == '-': initial_phase.append(pyspedas.time_double(-9999))
            else: initial_phase.append(pyspedas.time_double(column[col_ind+5]))
            main_phase.append(pyspedas.time_double(column[col_ind+10]))
            min_symh.append(pyspedas.time_double(column[col_ind+15]))
            end_time.append(pyspedas.time_double(column[col_ind+20]))
            symh_min_val.append(int(column[-1]))

del storm_id[0]    

phase_name = ['Full', 'Initial', 'Main', 'Recovery']

# %% Importing Updated List of geomagnetic storms

list_path = 'C:/Users/tawh2779/OneDrive/Documents/CU/Research/EMIC_All Storms/Initial Phase Study/Phase Times.csv' #LASP computer

initial_phase = []
main_phase = []
min_symh = []
end_time = []

with open(list_path, 'r') as data:
    csv_reader = csv.reader(data, delimiter=',')
    for i in range(1): #number of header lines
        data.readline() #skip header lines
    for column in csv_reader:
        if any(column) == False: continue #skip blank lines
        initial_phase.append(pyspedas.time_double(column[0]))
        main_phase.append(pyspedas.time_double(column[1]))
        min_symh.append(pyspedas.time_double(column[2]))
        end_time.append(pyspedas.time_double(column[3]))
        
# %% Creating all spectrograms

storm_day = ''
before = 6 #number of hours before the storm to include; 0 = start at the beginning of the initial phase
after = 0 #number of hours after the storm to include; 0 = stop at the end of the recovery phase

for storm in range(607,686): #607 is 20150907 and 685 is 20190930
    if initial_phase[storm-1] == -9999: continue
    else: 
        time_range = [initial_phase[storm-1]-(before*3600),end_time[storm-1]+(after*3600)]
        
    if type(time_range[0]) == str: time_range = pyspedas.time_double(time_range)

    if pyspedas.time_string(initial_phase[storm-1])[0:10] == storm_day: storm_day = pyspedas.time_string(initial_phase[storm-1])[0:10]+'_Storm2'
    else: storm_day = pyspedas.time_string(initial_phase[storm-1])[0:10]

    phase_times = [initial_phase[storm-1], main_phase[storm-1], min_symh[storm-1], end_time[storm-1]]

    # # %% Loading data
        # Loading Van Allen Probes
    rbspa_var = pyspedas.rbsp.emfisis(trange=time_range, time_clip=True, probe='a', coord='gsm', cadence='hires')
    rbspa_mag = pytplot.get_data('Magnitude') # Magnitude is the combination of each of the three Mag components, nT
    rbspa_mag[1][np.where(rbspa_mag[1] <= -6.55e4)[0]] = np.nan # from CDAWeb metadata, valid min is -6.55e4
    rbspa_mag[1][np.where(rbspa_mag[1] >= 6.55e4)[0]] = np.nan
    rbspa_coord = pytplot.get_data('coordinates') # GSM position vector, km
    rbspa_ephem_var = pyspedas.projects.rbsp.magephem(trange=time_range, probe='a', time_clip=True, cadence='1min', coord='t89q')
    rbspa_l = pytplot.get_data('Lsimple') # L-shell
    rbspa_imag = np.ones(len(rbspa_l[0])) # value of 1 is where the spacecraft is in the inner magnetosphere (0 is outside)
    rbspa_mlt = pytplot.get_data('CDMAG_MLT')
    
    rbspb_var = pyspedas.rbsp.emfisis(trange=time_range, time_clip=True, probe='b', coord='gsm', cadence='hires')
    rbspb_mag = pytplot.get_data('Magnitude')
    rbspb_mag[1][np.where(rbspb_mag[1] <= -6.55e4)[0]] = np.nan
    rbspb_mag[1][np.where(rbspb_mag[1] >= 6.55e4)[0]] = np.nan
    rbspb_coord = pytplot.get_data('coordinates')
    rbspb_ephem_var = pyspedas.projects.rbsp.magephem(trange=time_range, probe='b', time_clip=True, cadence='1min', coord='t89q')
    rbspb_l = pytplot.get_data('Lsimple')
    rbspb_imag = np.ones(len(rbspb_l[0]))
    rbspb_mlt = pytplot.get_data('CDMAG_MLT')

    # Loading MMS
    mms1_var = pyspedas.projects.mms.mms_load_fgm(trange=time_range, time_clip=True, probe=1)
    mms1_mag = pytplot.get_data('mms1_fgm_b_gsm_srvy_l2_btot')
    mms1_mag[1][np.where(mms1_mag[1] <= -1.7e4)[0]] = np.nan # from CDAWeb metadata, valid min is -1.7e4
    mms1_mag[1][np.where(mms1_mag[1] >= 1.7e4)[0]] = np.nan
    mms1_coord_var = pyspedas.projects.mms.mms_load_mec(trange=time_range, time_clip=True, probe=1) # MEC = magnetic ephemeris and coordinates
    mms1_coord = pytplot.get_data('mms1_mec_r_gsm') # GSM position vector, km
    mms1_l = pytplot.get_data('mms1_mec_l_dipole') # dipole L-shell
    mms1_imag = pytplot.get_data('mms1_mec_fieldline_type')
    mms1_mlt = pytplot.get_data('mms1_mec_mlt')

    mms2_var = pyspedas.projects.mms.mms_load_fgm(trange=time_range, time_clip=True, probe=2)
    mms2_mag = pytplot.get_data('mms2_fgm_b_gsm_srvy_l2_btot')
    mms2_mag[1][np.where(mms2_mag[1] <= -1.7e4)[0]] = np.nan 
    mms2_mag[1][np.where(mms2_mag[1] >= 1.7e4)[0]] = np.nan
    mms2_coord_var = pyspedas.projects.mms.mms_load_mec(trange=time_range, time_clip=True, probe=2)
    mms2_coord = pytplot.get_data('mms2_mec_r_gsm')
    mms2_l = pytplot.get_data('mms2_mec_l_dipole')
    mms2_imag = pytplot.get_data('mms2_mec_fieldline_type')
    mms2_mlt = pytplot.get_data('mms2_mec_mlt')

    mms3_var = pyspedas.projects.mms.mms_load_fgm(trange=time_range, time_clip=True, probe=3)
    mms3_mag = pytplot.get_data('mms3_fgm_b_gsm_srvy_l2_btot')
    mms3_mag[1][np.where(mms3_mag[1] <= -1.7e4)[0]] = np.nan 
    mms3_mag[1][np.where(mms3_mag[1] >= 1.7e4)[0]] = np.nan
    mms3_coord_var = pyspedas.projects.mms.mms_load_mec(trange=time_range, time_clip=True, probe=3)
    mms3_coord = pytplot.get_data('mms3_mec_r_gsm')
    mms3_l = pytplot.get_data('mms3_mec_l_dipole')
    mms3_imag = pytplot.get_data('mms3_mec_fieldline_type')
    mms3_mlt = pytplot.get_data('mms3_mec_mlt')

    mms4_var = pyspedas.projects.mms.mms_load_fgm(trange=time_range, time_clip=True, probe=4)
    mms4_mag = pytplot.get_data('mms4_fgm_b_gsm_srvy_l2_btot')
    mms4_mag[1][np.where(mms4_mag[1] <= -1.7e4)[0]] = np.nan 
    mms4_mag[1][np.where(mms4_mag[1] >= 1.7e4)[0]] = np.nan
    mms4_coord_var = pyspedas.projects.mms.mms_load_mec(trange=time_range, time_clip=True, probe=4)
    mms4_coord = pytplot.get_data('mms4_mec_r_gsm')
    mms4_l = pytplot.get_data('mms4_mec_l_dipole')
    mms4_imag = pytplot.get_data('mms4_mec_fieldline_type')
    mms4_mlt = pytplot.get_data('mms4_mec_mlt')

    # Loading THEMIS
    tha_var = pyspedas.themis.fgm(trange=time_range, time_clip=True, probe='a', coord='gsm')
    tha_mag = pytplot.get_data('tha_fgl_btotal')
    tha_mag[1][np.where(tha_mag[1] <= 0)[0]] = np.nan # from CDAWeb metadata, valid min is 0
    tha_mag[1][np.where(tha_mag[1] >= 2.5e4)[0]] = np.nan
    tha_coord_var = pyspedas.projects.themis.ssc(trange=time_range, time_clip=True, probe='a')
    tha_coord = pytplot.get_data('XYZ_GSM')
    tha_l = pytplot.get_data('L_VALUE')
    tha_dmagpause = pytplot.get_data('MAG_PAUSE')
    tha_mlt = pytplot.get_data('SM_LCT_T')

    thd_var = pyspedas.themis.fgm(trange=time_range, time_clip=True, probe='d', coord='gsm')
    thd_mag = pytplot.get_data('thd_fgl_btotal')
    thd_mag[1][np.where(thd_mag[1] <= 0)[0]] = np.nan
    thd_mag[1][np.where(thd_mag[1] >= 2.5e4)[0]] = np.nan
    thd_coord_var = pyspedas.projects.themis.ssc(trange=time_range, time_clip=True, probe='d')
    thd_coord = pytplot.get_data('XYZ_GSM') # GSM position vector, Re
    thd_l = pytplot.get_data('L_VALUE') # L-shell
    thd_dmagpause = pytplot.get_data('MAG_PAUSE') # distance to RS93 magnetopause, Re
    thd_mlt = pytplot.get_data('SM_LCT_T')

    the_var = pyspedas.themis.fgm(trange=time_range, time_clip=True, probe='e', coord='gsm')
    the_mag = pytplot.get_data('the_fgl_btotal')
    the_mag[1][np.where(the_mag[1] <= 0)[0]] = np.nan
    the_mag[1][np.where(the_mag[1] >= 2.5e4)[0]] = np.nan
    the_coord_var = pyspedas.projects.themis.ssc(trange=time_range, time_clip=True, probe='e')
    the_coord = pytplot.get_data('XYZ_GSM')
    the_l = pytplot.get_data('L_VALUE')
    the_dmagpause = pytplot.get_data('MAG_PAUSE')
    the_mlt = pytplot.get_data('SM_LCT_T')

    # Loading GOES
    g15_var = pyspedas.projects.goes.fgm(trange=time_range, time_clip=True, probe='15', datatype='512ms', instrument='fgm')
    g15_mag = pytplot.get_data('g15_fgm_BTSC_1') # spacecraft coordinates, nT
    g15_mag[1][np.where(g15_mag[1] <= -1024)[0]] = np.nan # from CDAWeb metadata, valid min is -1024
    g15_mag[1][np.where(g15_mag[1] >= 1024)[0]] = np.nan
    g15_coord_var = pyspedas.projects.goes.orbit(trange=time_range, time_clip=True, probe='15')
    g15_coord = pytplot.get_data('g15_orbit_XYZ_GSM') # GSM position vector, Re
    g15_l = pytplot.get_data('g15_orbit_L_VALUE') # L-shell
    g15_dmagpause = pytplot.get_data('g15_orbit_MAG_PAUSE') # distance to RS93 magnetopause, Re
    g15_mlt = pytplot.get_data('g15_orbit_SM_LCT_T')

    # # %% Calculating spectrograms
    
    # Van Allen
    rbsp_fs = 64 #Hz, sampling frequency
    rbsp_window = 1024 # spectrogram window size
    rbsp_avg_fs = rbsp_fs*60 # to average the fcp data over one minute
    rbspa_fcp_arr = fcp(rbspa_mag[1], rbsp_avg_fs)
    
    rbspa_freq, rbspa_t, rbspa_spec = spec(rbspa_mag, rbsp_fs, rbsp_window) #converting to spectrogram data
    pytplot.store_data('rbspa_dpwrspc', data={'x': rbspa_t, 'y': rbspa_spec, 'v': rbspa_freq}) #storing as a pytplot variable
    
    pytplot.store_data('rbspa_fcp', data={'x': rbspa_mag[0][::rbsp_avg_fs], 'y': rbspa_fcp_arr}) #storing as a pytplot variable
    pytplot.store_data('rbspa_fcHe', data={'x': rbspa_mag[0][::rbsp_avg_fs], 'y': rbspa_fcp_arr/4}) #Helium gyrofrequency
    pytplot.store_data('rbspa_fcO', data={'x': rbspa_mag[0][::rbsp_avg_fs], 'y': rbspa_fcp_arr/16}) #Oxygen gyrofrequency
    pytplot.store_data('rbspa_plotting', data=['rbspa_dpwrspc', 'rbspa_fcp', 'rbspa_fcHe', 'rbspa_fcO'])
    
    #saving the data used so it doesn't have to re-load for future runs
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-mag', rbspa_mag)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-time', rbspa_coord[0])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-coord', rbspa_coord[1])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-l', rbspa_l)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-imag', rbspa_imag)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-mlt', rbspa_mlt)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-freq', rbspa_freq)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-t', rbspa_t)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-spec', rbspa_spec)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-fcp', rbspa_fcp_arr)

    rbspb_fcp_arr = fcp(rbspb_mag[1], rbsp_avg_fs)
    
    rbspb_freq, rbspb_t, rbspb_spec = spec(rbspb_mag, rbsp_fs, rbsp_window)
    pytplot.store_data('rbspb_dpwrspc', data={'x': rbspb_t, 'y': rbspb_spec, 'v': rbspb_freq})
    
    pytplot.store_data('rbspb_fcp', data={'x': rbspb_mag[0][::rbsp_avg_fs], 'y': rbspb_fcp_arr}) #storing as a pytplot variable
    pytplot.store_data('rbspb_fcHe', data={'x': rbspb_mag[0][::rbsp_avg_fs], 'y': rbspb_fcp_arr/4}) #Helium gyrofrequency
    pytplot.store_data('rbspb_fcO', data={'x': rbspb_mag[0][::rbsp_avg_fs], 'y': rbspb_fcp_arr/16}) #Oxygen gyrofrequency
    pytplot.store_data('rbspb_plotting', data=['rbspb_dpwrspc', 'rbspb_fcp', 'rbspb_fcHe', 'rbspb_fcO'])
    
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-mag', rbspb_mag)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-time', rbspb_coord[0])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-coord', rbspb_coord[1])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-l', rbspb_l)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-imag', rbspb_imag)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-mlt', rbspb_mlt)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-freq', rbspb_freq)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-t', rbspb_t)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-spec', rbspb_spec)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-fcp', rbspb_fcp_arr)
    
    # MMS
    mms_fs = 8 #8 Hz for slow survey mode data; 16 Hz for fast survey mode data
    mms_window = 128
    mms_avg_fs = mms_fs*60
    mms1_fcp_arr = fcp(mms1_mag[1], mms_avg_fs)
    
    mms1_freq, mms1_t, mms1_spec = spec(mms1_mag, mms_fs, mms_window)
    pytplot.store_data('mms1_dpwrspc', data={'x': mms1_t, 'y': mms1_spec, 'v': mms1_freq})
    
    pytplot.store_data('mms1_fcp', data={'x': mms1_mag[0][::mms_avg_fs], 'y': mms1_fcp_arr}) #storing as a pytplot variable
    pytplot.store_data('mms1_fcHe', data={'x': mms1_mag[0][::mms_avg_fs], 'y': mms1_fcp_arr/4}) #Helium gyrofrequency
    pytplot.store_data('mms1_fcO', data={'x': mms1_mag[0][::mms_avg_fs], 'y': mms1_fcp_arr/16}) #Oxygen gyrofrequency
    pytplot.store_data('mms1_plotting', data=['mms1_dpwrspc', 'mms1_fcp', 'mms1_fcHe', 'mms1_fcO'])
    
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms1-mag', mms1_mag)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms1-time', mms1_coord[0])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms1-coord', mms1_coord[1])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms1-l', mms1_l)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms1-imag', mms1_imag)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms1-mlt', mms1_mlt)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms1-freq', mms1_freq)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms1-t', mms1_t)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms1-spec', mms1_spec)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms1-fcp', mms1_fcp_arr)
    
    mms2_fcp_arr = fcp(mms2_mag[1], mms_avg_fs)
    
    mms2_freq, mms2_t, mms2_spec = spec(mms2_mag, mms_fs, mms_window)
    pytplot.store_data('mms2_dpwrspc', data={'x': mms2_t, 'y': mms2_spec, 'v': mms2_freq})
    
    pytplot.store_data('mms2_fcp', data={'x': mms2_mag[0][::mms_avg_fs], 'y': mms2_fcp_arr}) #storing as a pytplot variable
    pytplot.store_data('mms2_fcHe', data={'x': mms2_mag[0][::mms_avg_fs], 'y': mms2_fcp_arr/4}) #Helium gyrofrequency
    pytplot.store_data('mms2_fcO', data={'x': mms2_mag[0][::mms_avg_fs], 'y': mms2_fcp_arr/16}) #Oxygen gyrofrequency
    pytplot.store_data('mms2_plotting', data=['mms2_dpwrspc', 'mms2_fcp', 'mms2_fcHe', 'mms2_fcO'])
    
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms2-mag', mms2_mag)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms2-time', mms2_coord[0])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms2-coord', mms2_coord[1])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms2-l', mms2_l)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms2-imag', mms2_imag)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms2-mlt', mms2_mlt)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms2-freq', mms2_freq)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms2-t', mms2_t)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms2-spec', mms2_spec)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms2-fcp', mms2_fcp_arr)
    
    mms3_fcp_arr = fcp(mms3_mag[1], mms_avg_fs)
    
    mms3_freq, mms3_t, mms3_spec = spec(mms3_mag, mms_fs, mms_window)
    pytplot.store_data('mms3_dpwrspc', data={'x': mms3_t, 'y': mms3_spec, 'v': mms3_freq})
    
    pytplot.store_data('mms3_fcp', data={'x': mms3_mag[0][::mms_avg_fs], 'y': mms3_fcp_arr}) #storing as a pytplot variable
    pytplot.store_data('mms3_fcHe', data={'x': mms3_mag[0][::mms_avg_fs], 'y': mms3_fcp_arr/4}) #Helium gyrofrequency
    pytplot.store_data('mms3_fcO', data={'x': mms3_mag[0][::mms_avg_fs], 'y': mms3_fcp_arr/16}) #Oxygen gyrofrequency
    pytplot.store_data('mms3_plotting', data=['mms3_dpwrspc', 'mms3_fcp', 'mms3_fcHe', 'mms3_fcO'])
    
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms3-mag', mms3_mag)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms3-time', mms3_coord[0])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms3-coord', mms3_coord[1])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms3-l', mms3_l)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms3-imag', mms3_imag)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms3-mlt', mms3_mlt)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms3-freq', mms3_freq)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms3-t', mms3_t)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms3-spec', mms3_spec)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms3-fcp', mms3_fcp_arr)
    
    mms4_fcp_arr = fcp(mms4_mag[1], mms_avg_fs)
    
    mms4_freq, mms4_t, mms4_spec = spec(mms4_mag, mms_fs, mms_window)
    pytplot.store_data('mms4_dpwrspc', data={'x': mms4_t, 'y': mms4_spec, 'v': mms4_freq})
    
    pytplot.store_data('mms4_fcp', data={'x': mms4_mag[0][::mms_avg_fs], 'y': mms4_fcp_arr}) #storing as a pytplot variable
    pytplot.store_data('mms4_fcHe', data={'x': mms4_mag[0][::mms_avg_fs], 'y': mms4_fcp_arr/4}) #Helium gyrofrequency
    pytplot.store_data('mms4_fcO', data={'x': mms4_mag[0][::mms_avg_fs], 'y': mms4_fcp_arr/16}) #Oxygen gyrofrequency
    pytplot.store_data('mms4_plotting', data=['mms4_dpwrspc', 'mms4_fcp', 'mms4_fcHe', 'mms4_fcO'])
    
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms4-mag', mms4_mag)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms4-time', mms4_coord[0])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms4-coord', mms4_coord[1])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms4-l', mms4_l)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms4-imag', mms4_imag)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms4-mlt', mms4_mlt)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms4-freq', mms4_freq)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms4-t', mms4_t)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms4-spec', mms4_spec)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_mms4-fcp', mms4_fcp_arr)
    
    # THEMIS
    th_fs = 4
    th_window = 256
    th_avg_fs = th_fs*60
    thd_fcp_arr = fcp(thd_mag[1], th_avg_fs)
    
    thd_freq, thd_t, thd_spec = spec(thd_mag, th_fs, th_window)
    pytplot.store_data('thd_dpwrspc', data={'x': thd_t, 'y': thd_spec, 'v': thd_freq})
    
    pytplot.store_data('thd_fcp', data={'x': thd_mag[0][::th_avg_fs], 'y': thd_fcp_arr}) #storing as a pytplot variable
    pytplot.store_data('thd_fcHe', data={'x': thd_mag[0][::th_avg_fs], 'y': thd_fcp_arr/4}) #Helium gyrofrequency
    pytplot.store_data('thd_fcO', data={'x': thd_mag[0][::th_avg_fs], 'y': thd_fcp_arr/16}) #Oxygen gyrofrequency
    pytplot.store_data('thd_plotting', data=['thd_dpwrspc', 'thd_fcp', 'thd_fcHe', 'thd_fcO'])
    
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_thd-mag', thd_mag)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_thd-time', thd_coord[0])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_thd-coord', thd_coord[1])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_thd-l', thd_l)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_thd-dmagpause', thd_dmagpause)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_thd-mlt', thd_mlt)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_thd-freq', thd_freq)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_thd-t', thd_t)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_thd-spec', thd_spec)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_thd-fcp', thd_fcp_arr)
    
    tha_fcp_arr = fcp(tha_mag[1], th_avg_fs)
    
    tha_freq, tha_t, tha_spec = spec(tha_mag, th_fs, th_window)
    pytplot.store_data('tha_dpwrspc', data={'x': tha_t, 'y': tha_spec, 'v': tha_freq})
    
    pytplot.store_data('tha_fcp', data={'x': tha_mag[0][::th_avg_fs], 'y': tha_fcp_arr}) #storing as a pytplot variable
    pytplot.store_data('tha_fcHe', data={'x': tha_mag[0][::th_avg_fs], 'y': tha_fcp_arr/4}) #Helium gyrofrequency
    pytplot.store_data('tha_fcO', data={'x': tha_mag[0][::th_avg_fs], 'y': tha_fcp_arr/16}) #Oxygen gyrofrequency
    pytplot.store_data('tha_plotting', data=['tha_dpwrspc', 'tha_fcp', 'tha_fcHe', 'tha_fcO'])
    
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_tha-mag', tha_mag)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_tha-time', tha_coord[0])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_tha-coord', tha_coord[1])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_tha-l', tha_l)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_tha-dmagpause', tha_dmagpause)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_tha-mlt', tha_mlt)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_tha-freq', tha_freq)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_tha-t', tha_t)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_tha-spec', tha_spec)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_tha-fcp', tha_fcp_arr)
    
    the_fcp_arr = fcp(the_mag[1], th_avg_fs)
    
    the_freq, the_t, the_spec = spec(the_mag, th_fs, th_window)
    pytplot.store_data('the_dpwrspc', data={'x': the_t, 'y': the_spec, 'v': the_freq})
    
    pytplot.store_data('the_fcp', data={'x': the_mag[0][::th_avg_fs], 'y': the_fcp_arr}) #storing as a pytplot variable
    pytplot.store_data('the_fcHe', data={'x': the_mag[0][::th_avg_fs], 'y': the_fcp_arr/4}) #Helium gyrofrequency
    pytplot.store_data('the_fcO', data={'x': the_mag[0][::th_avg_fs], 'y': the_fcp_arr/16}) #Oxygen gyrofrequency
    pytplot.store_data('the_plotting', data=['the_dpwrspc', 'the_fcp', 'the_fcHe', 'the_fcO'])
    
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_the-mag', the_mag)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_the-time', the_coord[0])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_the-coord', the_coord[1])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_the-l', the_l)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_the-dmagpause', the_dmagpause)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_the-mlt', the_mlt)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_the-freq', the_freq)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_the-t', the_t)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_the-spec', the_spec)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_the-fcp', the_fcp_arr)
    
    # GOES
    goes_fs = 1/0.512
    goes_window = 32
    goes_avg_fs = 2 * 60 #approximately 1-minute average (fs is 1/0.512 instead of 2)
    g15_fcp_arr = fcp(g15_mag[1], goes_avg_fs)
    
    g15_freq, g15_t, g15_spec = spec(g15_mag, goes_fs, goes_window)
    pytplot.store_data('g15_dpwrspc', data={'x': g15_t, 'y': g15_spec, 'v': g15_freq})
        
    pytplot.store_data('g15_fcp', data={'x': g15_mag[0][::goes_avg_fs], 'y': g15_fcp_arr}) #storing as a pytplot variable
    pytplot.store_data('g15_fcHe', data={'x': g15_mag[0][::goes_avg_fs], 'y': g15_fcp_arr/4}) #Helium gyrofrequency
    pytplot.store_data('g15_fcO', data={'x': g15_mag[0][::goes_avg_fs], 'y': g15_fcp_arr/16}) #Oxygen gyrofrequency
    pytplot.store_data('g15_plotting', data=['g15_dpwrspc', 'g15_fcp', 'g15_fcHe', 'g15_fcO'])#, 'g15_vline'])
    
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_g15-mag', g15_mag)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_g15-time', g15_coord[0])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_g15-coord', g15_coord[1])
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_g15-l', g15_l)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_g15-dmagpause', g15_dmagpause)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_g15-mlt', g15_mlt)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_g15-freq', g15_freq)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_g15-t', g15_t)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_g15-spec', g15_spec)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_g15-fcp', g15_fcp_arr)
    
    # # %% Plotting OMNI Data

    if type(time_range[0]) == str: time_range_day = [time_range[0][0:10]+' 00:00', time_range[1][0:10]+' 23:59']
    elif type(time_range[0]) == float: time_range_day = [pyspedas.time_string(time_range[0])[0:10]+' 00:00', pyspedas.time_string(time_range[1])[0:10]+' 23:59']

    if np.abs(pyspedas.time_float(time_range[0]) - pyspedas.time_float(time_range_day[0])) < 6*3600:
        time_range_day[0] = pyspedas.time_string(pyspedas.time_float(time_range_day[0]) - 24*3600)
    if np.abs(pyspedas.time_float(time_range[1]) - pyspedas.time_float(time_range_day[1])) < 6*3600:
        time_range_day[1] = pyspedas.time_string(pyspedas.time_float(time_range_day[1]) + 24*3600)

    omni_var = pyspedas.projects.omni.data(trange=time_range_day)

    Bz = pytplot.get_data('BZ_GSM')
    Bx = pytplot.get_data('BX_GSE')
    By = pytplot.get_data('BY_GSM')
    Dp = pytplot.get_data('Pressure')
    symh = pytplot.get_data('SYM_H')
    speed = pytplot.get_data('flow_speed')
    density = pytplot.get_data('proton_density')
    AE = pytplot.get_data('AE_INDEX')
    Vx = pytplot.get_data('Vx')
    Vy = pytplot.get_data('Vy')
    Vz = pytplot.get_data('Vz')
    E = pytplot.get_data('E')

    pytplot.tplot_options('title', storm_day+' Solar Wind Data (GSM)')
    pytplot.tplot_options('xmargin', [0.15,0.05])
    pytplot.tplot_options('vertical_spacing', 0.2)
    pytplot.options('SYM_H', 'ytitle', 'SYM-H')
    pytplot.options('AE_INDEX', 'ytitle', 'AE')
    pytplot.options('BX_GSE', 'ytitle', 'Bx') #GSM?
    pytplot.options('BY_GSM', 'ytitle', 'By')
    pytplot.options('BZ_GSM', 'ytitle', 'Bz')
    pytplot.options('flow_speed', 'ytitle', 'Flow\nSpeed')
    pytplot.options('T', 'ytitle', 'Temperature')
    pytplot.options('Pressure', 'ytitle', 'Pressure')
    pytplot.options('proton_density', 'ytitle', 'Proton\nDensity')
    pyspedas.xlim(time_range_day[0], time_range_day[1])
    pytplot.timebar(phase_times, color=(0,128,128), thick=2)
    pytplot.tplot(['SYM_H', 'AE_INDEX', 'BZ_GSM', 'flow_speed', 'Pressure', 'proton_density'], \
                  dpi=150, save_png=file_path+storm_day+'_OMNI data_poster')
    # pytplot.tplot(['SYM_H', 'BX_GSE', 'BY_GSM', 'BZ_GSM', 'flow_speed', 'T', 'Pressure', 'proton_density'], \
    #               dpi=150, save_png=file_path+storm_day+'_OMNI data')

    np.save(file_path+'Satellite Storm Data/'+storm_day+'_OMNI_Bz', Bz)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_OMNI_Bx', Bx)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_OMNI_By', By)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_OMNI_Dp', Dp)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_OMNI_SYM-H', symh)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_OMNI_V', speed)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_OMNI_rho', density)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_OMNI_AE', AE)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_OMNI_Vx-GSE', Vx)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_OMNI_Vy-GSE', Vy)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_OMNI_Vz-GSE', Vz)
    np.save(file_path+'Satellite Storm Data/'+storm_day+'_OMNI_E', E)

    # %% Cycling through phases, with downloaded data; can instead incorporate into previous loop (skip to line 934) to load and go through the algorithm all at once
    
storm_day = ''
before = 6 #number of hours before the storm to include; 0 = start at the beginning of the initial phase
after = 0 #number of hours after the storm to include; 0 = stop at the end of the recovery phase

for storm in range(607,686): #607 is 20150907 and 685 is 20190930
    if initial_phase[storm-1] == -9999: continue
    else: 
        time_range = [initial_phase[storm-1]-(before*3600),initial_phase[storm-1]+(after*3600)]
        
    if type(time_range[0]) == str: time_range = pyspedas.time_double(time_range)

    if pyspedas.time_string(initial_phase[storm-1])[0:10] == storm_day: storm_day = pyspedas.time_string(initial_phase[storm-1])[0:10]+'_Storm2'
    else: storm_day = pyspedas.time_string(initial_phase[storm-1])[0:10]
    
    phase_times = [initial_phase[storm-1], main_phase[storm-1], min_symh[storm-1], end_time[storm-1]]

    rbspa_mag = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-mag.npy')
    rbspa_time_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-time.npy')
    rbspa_loc_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-coord.npy')
    rbspa_coord = [rbspa_time_coord, rbspa_loc_coord]
    rbspa_l = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-l.npy')
    rbspa_imag = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-imag.npy')
    rbspa_mlt = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-mlt.npy')
    rbspa_freq = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-freq.npy')
    rbspa_t = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-t.npy')
    rbspa_spec = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-spec.npy')
    rbspa_fcp_arr = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspa-fcp.npy')
    
    rbspb_mag = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-mag.npy')
    rbspb_time_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-time.npy')
    rbspb_loc_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-coord.npy')
    rbspb_coord = [rbspb_time_coord, rbspb_loc_coord]
    rbspb_l = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-l.npy')
    rbspb_imag = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-imag.npy')
    rbspb_mlt = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-mlt.npy')
    rbspb_freq = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-freq.npy')
    rbspb_t = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-t.npy')
    rbspb_spec = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-spec.npy')
    rbspb_fcp_arr = np.load(file_path+'Satellite Storm Data/'+storm_day+'_rbspb-fcp.npy')
    
    mms1_mag = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms1-mag.npy')
    mms1_time_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms1-time.npy')
    mms1_loc_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms1-coord.npy')
    mms1_coord = [mms1_time_coord, mms1_loc_coord]
    mms1_l = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms1-l.npy')
    mms1_imag = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms1-imag.npy')
    mms1_mlt = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms1-mlt.npy')
    mms1_freq = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms1-freq.npy')
    mms1_t = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms1-t.npy')
    mms1_spec = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms1-spec.npy')
    mms1_fcp_arr = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms1-fcp.npy')
    
    mms2_mag = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms2-mag.npy')
    mms2_time_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms2-time.npy')
    mms2_loc_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms2-coord.npy')
    mms2_coord = [mms2_time_coord, mms2_loc_coord]
    mms2_l = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms2-l.npy')
    mms2_imag = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms2-imag.npy')
    mms2_mlt = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms2-mlt.npy')
    mms2_freq = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms2-freq.npy')
    mms2_t = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms2-t.npy')
    mms2_spec = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms2-spec.npy')
    mms2_fcp_arr = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms2-fcp.npy')
    
    mms3_mag = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms3-mag.npy')
    mms3_time_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms3-time.npy')
    mms3_loc_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms3-coord.npy')
    mms3_coord = [mms3_time_coord, mms3_loc_coord]
    mms3_l = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms3-l.npy')
    mms3_imag = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms3-imag.npy')
    mms3_mlt = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms3-mlt.npy')
    mms3_freq = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms3-freq.npy')
    mms3_t = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms3-t.npy')
    mms3_spec = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms3-spec.npy')
    mms3_fcp_arr = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms3-fcp.npy')
    
    mms4_mag = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms4-mag.npy')
    mms4_time_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms4-time.npy')
    mms4_loc_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms4-coord.npy')
    mms4_coord = [mms4_time_coord, mms4_loc_coord]
    mms4_l = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms4-l.npy')
    mms4_imag = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms4-imag.npy')
    mms4_mlt = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms4-mlt.npy')
    mms4_freq = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms4-freq.npy')
    mms4_t = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms4-t.npy')
    mms4_spec = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms4-spec.npy')
    mms4_fcp_arr = np.load(file_path+'Satellite Storm Data/'+storm_day+'_mms4-fcp.npy')
    
    tha_mag = np.load(file_path+'Satellite Storm Data/'+storm_day+'_tha-mag.npy')
    tha_time_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_tha-time.npy')
    tha_loc_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_tha-coord.npy')
    tha_coord = [tha_time_coord, tha_loc_coord]
    tha_l = np.load(file_path+'Satellite Storm Data/'+storm_day+'_tha-l.npy')
    tha_dmagpause = np.load(file_path+'Satellite Storm Data/'+storm_day+'_tha-dmagpause.npy')
    tha_mlt = np.load(file_path+'Satellite Storm Data/'+storm_day+'_tha-mlt.npy')
    tha_freq = np.load(file_path+'Satellite Storm Data/'+storm_day+'_tha-freq.npy')
    tha_t = np.load(file_path+'Satellite Storm Data/'+storm_day+'_tha-t.npy')
    tha_spec = np.load(file_path+'Satellite Storm Data/'+storm_day+'_tha-spec.npy')
    tha_fcp_arr = np.load(file_path+'Satellite Storm Data/'+storm_day+'_tha-fcp.npy')
    
    thd_mag = np.load(file_path+'Satellite Storm Data/'+storm_day+'_thd-mag.npy')
    thd_time_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_thd-time.npy')
    thd_loc_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_thd-coord.npy')
    thd_coord = [thd_time_coord, thd_loc_coord]
    thd_l = np.load(file_path+'Satellite Storm Data/'+storm_day+'_thd-l.npy')
    thd_dmagpause = np.load(file_path+'Satellite Storm Data/'+storm_day+'_thd-dmagpause.npy')
    thd_mlt = np.load(file_path+'Satellite Storm Data/'+storm_day+'_thd-mlt.npy')
    thd_freq = np.load(file_path+'Satellite Storm Data/'+storm_day+'_thd-freq.npy')
    thd_t = np.load(file_path+'Satellite Storm Data/'+storm_day+'_thd-t.npy')
    thd_spec = np.load(file_path+'Satellite Storm Data/'+storm_day+'_thd-spec.npy')
    thd_fcp_arr = np.load(file_path+'Satellite Storm Data/'+storm_day+'_thd-fcp.npy')
    
    the_mag = np.load(file_path+'Satellite Storm Data/'+storm_day+'_the-mag.npy')
    the_time_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_the-time.npy')
    the_loc_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_the-coord.npy')
    the_coord = [the_time_coord, the_loc_coord]
    the_l = np.load(file_path+'Satellite Storm Data/'+storm_day+'_the-l.npy')
    the_dmagpause = np.load(file_path+'Satellite Storm Data/'+storm_day+'_the-dmagpause.npy')
    the_mlt = np.load(file_path+'Satellite Storm Data/'+storm_day+'_the-mlt.npy')
    the_freq = np.load(file_path+'Satellite Storm Data/'+storm_day+'_the-freq.npy')
    the_t = np.load(file_path+'Satellite Storm Data/'+storm_day+'_the-t.npy')
    the_spec = np.load(file_path+'Satellite Storm Data/'+storm_day+'_the-spec.npy')
    the_fcp_arr = np.load(file_path+'Satellite Storm Data/'+storm_day+'_the-fcp.npy')
    
    g15_mag = np.load(file_path+'Satellite Storm Data/'+storm_day+'_g15-mag.npy')
    g15_time_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_g15-time.npy')
    g15_loc_coord = np.load(file_path+'Satellite Storm Data/'+storm_day+'_g15-coord.npy')
    g15_coord = [g15_time_coord, g15_loc_coord]
    g15_l = np.load(file_path+'Satellite Storm Data/'+storm_day+'_g15-l.npy')
    g15_dmagpause = np.load(file_path+'Satellite Storm Data/'+storm_day+'_g15-dmagpause.npy')
    g15_mlt = np.load(file_path+'Satellite Storm Data/'+storm_day+'_g15-mlt.npy')
    g15_freq = np.load(file_path+'Satellite Storm Data/'+storm_day+'_g15-freq.npy')
    g15_t = np.load(file_path+'Satellite Storm Data/'+storm_day+'_g15-t.npy')
    g15_spec = np.load(file_path+'Satellite Storm Data/'+storm_day+'_g15-spec.npy')
    g15_fcp_arr = np.load(file_path+'Satellite Storm Data/'+storm_day+'_g15-fcp.npy')
    
    Bz = np.load(file_path+'Satellite Storm Data/'+storm_day+'_OMNI_Bz.npy')
    Dp = np.load(file_path+'Satellite Storm Data/'+storm_day+'_OMNI_Dp.npy')
        
    for phase in range(4):
            #0 = full time, 1 = initial phase, 2 = main phase, 3 = recovery phase
        if phase > 1: continue #right now only looking at initial phases
        if initial_phase[storm-1] == -9999 and phase == 1: continue #skip initial phase if there isn't one
        if phase == 0: 
            trange = pyspedas.time_double(time_range)
            continue
        else: trange = phase_times[phase-1:phase+1] 
        trange_str = pyspedas.time_string(trange)
        storm_phase = phase_name[phase]
        # else: trange = time_range #use lines 944-946 instead of 941-943 if looking at pre-storm data
        # trange_str = pyspedas.time_string(time_range)
        # storm_phase = 'Pre-Storm'

        ## %% Plotting all median-normalized spectrograms

        mag_avg_x, mag_avg_y, mag_min_x, mag_min_y, mag_max_x, mag_max_y, mag_avg_r, mag_min_r, mag_max_r = avg_magpause(Bz[1], Dp[1], Bz[0])

        # Van Allen Probes
        rbsp_l_min = 2.5 #minimum l-shell for valid measurements
        rbsp_fs = 64
        rbsp_avg_fs = rbsp_fs*60 # to average the fcp data over one minute

        rbspa_coord_new = [[],[]]
        rbspa_coord_new[0] = np.copy(rbspa_coord[0])
        rbspa_coord_new[1] = rbspa_coord[1]/Re
        rbspa_l_new = time_align(rbspa_l[0], rbspa_t, rbspa_l[1])
        rbspa_imag_new = time_align(rbspa_l[0], rbspa_t, rbspa_imag)
        rbspa_imag_new[1][np.where(rbspa_l_new[1] < rbsp_l_min)[0]] = 0
        rbspa_spec_new = np.zeros(rbspa_spec.shape)*np.nan
        rbspa_spec_new[np.where(rbspa_imag_new[1] == 1)[0],:] = rbspa_spec[np.where(rbspa_imag_new[1] == 1)[0],:]
        rbspa_median, rbspa_std = median_extraction(rbspa_t, rbspa_freq, rbspa_spec_new)
        rbspa_mask = np.zeros(rbspa_spec.shape)*np.nan
        rbspa_mask[np.where(rbspa_imag_new[1] == 0)[0],:] = 1 #rbspa_freq
        pytplot.store_data('rbspa_dpwrspc_extracted', data={'x': rbspa_t, 'y': rbspa_spec/rbspa_median, 'v': rbspa_freq})
        pytplot.store_data('rbspa_imag_mask', data={'x': rbspa_t, 'y': rbspa_mask, 'v': rbspa_freq})
        pytplot.store_data('rbspa_fcp', data={'x': rbspa_mag[0][::rbsp_avg_fs], 'y': rbspa_fcp_arr}) #storing as a pytplot variable
        pytplot.store_data('rbspa_fcHe', data={'x': rbspa_mag[0][::rbsp_avg_fs], 'y': rbspa_fcp_arr/4}) #Helium gyrofrequency
        pytplot.store_data('rbspa_fcO', data={'x': rbspa_mag[0][::rbsp_avg_fs], 'y': rbspa_fcp_arr/16}) #Oxygen gyrofrequency
        pytplot.store_data('rbspa_plotting_extracted', data=['rbspa_dpwrspc_extracted', 'rbspa_fcp', 'rbspa_fcHe', 'rbspa_fcO', 'rbspa_imag_mask'])
        pytplot.options('rbspa_imag_mask', 'spec', 1)
        pytplot.options('rbspa_imag_mask', 'colormap', 'binary')
        pytplot.options('rbspa_imag_mask', 'alpha', 0.67)
        pytplot.options('rbspa_imag_mask', 'second_axis_size', 0.25)
        pytplot.options('rbspa_imag_mask', 'zrange', [0,1])
        pytplot.options('rbspa_dpwrspc_extracted', 'colormap', 'jet')
        pytplot.options('rbspa_dpwrspc_extracted', 'data_gap', 10*60) #number in seconds
        pytplot.options('rbspa_fcp', 'color', 'white')
        pytplot.options('rbspa_fcp', 'thick', 1)
        pytplot.options('rbspa_fcHe', 'color', 'white')
        pytplot.options('rbspa_fcHe', 'thick', 1)
        pytplot.options('rbspa_fcO', 'color', 'white')
        pytplot.options('rbspa_fcO', 'thick', 1)
        pytplot.options('rbspa_plotting_extracted', 'ytitle', 'RBSP\nA\n')
        pytplot.options('rbspa_plotting_extracted', 'ztitle', ' ')
        pytplot.options('rbspa_plotting_extracted', 'ylog', False)
        pytplot.options('rbspa_dpwrspc_extracted', 'zlog', True)
        pytplot.options('rbspa_imag_mask', 'zlog', False)
        pytplot.options('rbspa_dpwrspc_extracted', 'second_axis_size', 0)
        pytplot.options('rbspa_plotting_extracted', 'yrange', [0,6])
        pytplot.options('rbspa_dpwrspc_extracted', 'zrange', [1e-1, 1e4])
        pytplot.options('rbspa_dpwrspc_extracted', 'spec', 1)
        
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_rbspa-time_new', rbspa_coord_new[0])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_rbspa-coord_new', rbspa_coord_new[1])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_rbspa-l_new', rbspa_l_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_rbspa-imag', rbspa_imag_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_rbspa-spec_new', rbspa_spec_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_rbspa-spec_norm', rbspa_spec/rbspa_median)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_rbspa-mask', rbspa_mask)

        rbspb_coord_new = [[],[]]
        rbspb_coord_new[0] = np.copy(rbspb_coord[0])
        rbspb_coord_new[1] = rbspb_coord[1]/Re
        rbspb_l_new = time_align(rbspb_l[0], rbspb_t, rbspb_l[1])
        rbspb_imag_new = time_align(rbspb_l[0], rbspb_t, rbspb_imag)
        rbspb_imag_new[1][np.where(rbspb_l_new[1] < rbsp_l_min)[0]] = 0
        rbspb_spec_new = np.zeros(rbspb_spec.shape)*np.nan
        rbspb_spec_new[np.where(rbspb_imag_new[1] == 1)[0],:] = rbspb_spec[np.where(rbspb_imag_new[1] == 1)[0],:]
        rbspb_median, rbspb_std = median_extraction(rbspb_t, rbspb_freq, rbspb_spec_new)
        rbspb_mask = np.zeros(rbspb_spec.shape)*np.nan
        rbspb_mask[np.where(rbspb_imag_new[1] == 0)[0],:] = 1
        pytplot.store_data('rbspb_dpwrspc_extracted', data={'x': rbspb_t, 'y': rbspb_spec/rbspb_median, 'v': rbspb_freq})
        pytplot.store_data('rbspb_imag_mask', data={'x': rbspb_t, 'y': rbspb_mask, 'v': rbspb_freq})
        pytplot.store_data('rbspb_fcp', data={'x': rbspb_mag[0][::rbsp_avg_fs], 'y': rbspb_fcp_arr}) #storing as a pytplot variable
        pytplot.store_data('rbspb_fcHe', data={'x': rbspb_mag[0][::rbsp_avg_fs], 'y': rbspb_fcp_arr/4}) #Helium gyrofrequency
        pytplot.store_data('rbspb_fcO', data={'x': rbspb_mag[0][::rbsp_avg_fs], 'y': rbspb_fcp_arr/16}) #Oxygen gyrofrequency
        pytplot.store_data('rbspb_plotting_extracted', data=['rbspb_dpwrspc_extracted', 'rbspb_fcp', 'rbspb_fcHe', 'rbspb_fcO', 'rbspb_imag_mask'])
        pytplot.options('rbspb_imag_mask', 'spec', 1)
        pytplot.options('rbspb_imag_mask', 'colormap', 'binary')
        pytplot.options('rbspb_imag_mask', 'alpha', 0.67)
        pytplot.options('rbspb_imag_mask', 'second_axis_size', 0.25)
        pytplot.options('rbspb_imag_mask', 'zrange', [0,1])
        pytplot.options('rbspb_dpwrspc_extracted', 'colormap', 'jet')
        pytplot.options('rbspb_dpwrspc_extracted', 'data_gap', 10*60)
        pytplot.options('rbspb_fcp', 'color', 'white')
        pytplot.options('rbspb_fcp', 'thick', 1)
        pytplot.options('rbspb_fcHe', 'color', 'white')
        pytplot.options('rbspb_fcHe', 'thick', 1)
        pytplot.options('rbspb_fcO', 'color', 'white')
        pytplot.options('rbspb_fcO', 'thick', 1)
        pytplot.options('rbspb_plotting_extracted', 'ytitle', 'RBSP\nB\n')
        pytplot.options('rbspb_plotting_extracted', 'ztitle', ' ')
        pytplot.options('rbspb_plotting_extracted', 'ylog', False)
        pytplot.options('rbspb_dpwrspc_extracted', 'zlog', True)
        pytplot.options('rbspb_imag_mask', 'zlog', False)
        pytplot.options('rbspb_dpwrspc_extracted', 'second_axis_size', 0)
        pytplot.options('rbspb_plotting_extracted', 'yrange', [0,6])
        pytplot.options('rbspb_dpwrspc_extracted', 'zrange', [1e-1, 1e4])
        pytplot.options('rbspb_dpwrspc_extracted', 'spec', 1)

        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_rbspb-time_new', rbspb_coord_new[0])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_rbspb-coord_new', rbspb_coord_new[1])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_rbspb-l_new', rbspb_l_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_rbspb-imag', rbspb_imag_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_rbspb-spec_new', rbspb_spec_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_rbspb-spec_norm', rbspb_spec/rbspb_median)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_rbspb-mask', rbspb_mask)

        # MMS
        mms_l_min = 2.5 #minimum l-shell for valid measurements
        mms_fs = 8
        mms_avg_fs = mms_fs*60

        mms1_coord_new = time_align(mms1_coord[0], mms1_t, mms1_coord[1]/Re)
        mms1_l_new = time_align(mms1_l[0], mms1_t, mms1_l[1])
        mms1_imag_new = time_align(mms1_imag[0], mms1_t, mms1_imag[1])
        mms1_imag_new[1][np.where(mms1_imag_new[1] > 0)[0]] = 1
        mms1_imag_new[1] = magpause(mag_min_r, mms1_coord_new[1][:,0], mms1_coord_new[1][:,1], mms1_imag_new[1], mms1_l_new[1], mms_l_min)
        mms1_spec_new = np.zeros(mms1_spec.shape)*np.nan
        mms1_spec_new[np.where(mms1_imag_new[1] == 1)[0],:] = mms1_spec[np.where(mms1_imag_new[1] == 1)[0],:]
        mms1_median, mms1_std = median_extraction(mms1_t, mms1_freq, mms1_spec_new)
        mms1_mask = np.zeros(mms1_spec.shape)*np.nan
        mms1_mask[np.where(mms1_imag_new[1] == 0)[0],:] = 1
        pytplot.store_data('mms1_dpwrspc_extracted', data={'x': mms1_t, 'y': mms1_spec/mms1_median, 'v': mms1_freq})
        pytplot.store_data('mms1_imag_mask', data={'x': mms1_t, 'y': mms1_mask, 'v': mms1_freq})
        pytplot.store_data('mms1_fcp', data={'x': mms1_mag[0][::mms_avg_fs], 'y': mms1_fcp_arr}) #storing as a pytplot variable
        pytplot.store_data('mms1_fcHe', data={'x': mms1_mag[0][::mms_avg_fs], 'y': mms1_fcp_arr/4}) #Helium gyrofrequency
        pytplot.store_data('mms1_fcO', data={'x': mms1_mag[0][::mms_avg_fs], 'y': mms1_fcp_arr/16}) #Oxygen gyrofrequency
        pytplot.store_data('mms1_plotting_extracted', data=['mms1_dpwrspc_extracted', 'mms1_fcp', 'mms1_fcHe', 'mms1_fcO', 'mms1_imag_mask'])
        pytplot.options('mms1_imag_mask', 'spec', 1)
        pytplot.options('mms1_imag_mask', 'colormap', 'binary')
        pytplot.options('mms1_imag_mask', 'alpha', 0.67)
        pytplot.options('mms1_imag_mask', 'second_axis_size', 0.25)
        pytplot.options('mms1_imag_mask', 'zrange', [0,1])
        pytplot.options('mms1_dpwrspc_extracted', 'colormap', 'jet')
        pytplot.options('mms1_dpwrspc_extracted', 'data_gap', 10*60)
        pytplot.options('mms1_fcp', 'color', 'white')
        pytplot.options('mms1_fcp', 'thick', 1)
        pytplot.options('mms1_fcHe', 'color', 'white')
        pytplot.options('mms1_fcHe', 'thick', 1)
        pytplot.options('mms1_fcO', 'color', 'white')
        pytplot.options('mms1_fcO', 'thick', 1)
        pytplot.options('mms1_plotting_extracted', 'ytitle','MMS\n1\n')
        pytplot.options('mms1_plotting_extracted', 'ztitle',' ')
        pytplot.options('mms1_plotting_extracted', 'ylog', False)
        pytplot.options('mms1_dpwrspc_extracted', 'zlog', True)
        pytplot.options('mms1_imag_mask', 'zlog', False)
        pytplot.options('mms1_dpwrspc_extracted', 'second_axis_size', 0)
        pytplot.options('mms1_plotting_extracted', 'yrange', [0,4])
        pytplot.options('mms1_dpwrspc_extracted', 'zrange', [1e-1, 1e4])
        pytplot.options('mms1_dpwrspc_extracted', 'spec', 1)

        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms1-time_new', mms1_coord_new[0])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms1-coord_new', mms1_coord_new[1])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms1-l_new', mms1_l_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms1-imag', mms1_imag_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms1-spec_new', mms1_spec_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms1-spec_norm', mms1_spec/mms1_median)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms1-mask', mms1_mask)

        mms2_coord_new = time_align(mms2_coord[0], mms2_t, mms2_coord[1]/Re)
        mms2_l_new = time_align(mms2_l[0], mms2_t, mms2_l[1])
        mms2_imag_new = time_align(mms2_imag[0], mms2_t, mms2_imag[1])
        mms2_imag_new[1][np.where(mms2_imag_new[1] > 0)[0]] = 1
        mms2_imag_new[1] = magpause(mag_min_r, mms2_coord_new[1][:,0], mms2_coord_new[1][:,1], mms2_imag_new[1], mms2_l_new[1], mms_l_min)
        mms2_spec_new = np.zeros(mms2_spec.shape)*np.nan
        mms2_spec_new[np.where(mms2_imag_new[1] == 1)[0],:] = mms2_spec[np.where(mms2_imag_new[1] == 1)[0],:]
        mms2_median, mms2_std = median_extraction(mms2_t, mms2_freq, mms2_spec_new)
        mms2_mask = np.zeros(mms2_spec.shape)*np.nan
        mms2_mask[np.where(mms2_imag_new[1] == 0)[0],:] = 1
        pytplot.store_data('mms2_dpwrspc_extracted', data={'x': mms2_t, 'y': mms2_spec/mms2_median, 'v': mms2_freq})
        pytplot.store_data('mms2_imag_mask', data={'x': mms2_t, 'y': mms2_mask, 'v': mms2_freq})
        pytplot.store_data('mms2_fcp', data={'x': mms2_mag[0][::mms_avg_fs], 'y': mms2_fcp_arr}) #storing as a pytplot variable
        pytplot.store_data('mms2_fcHe', data={'x': mms2_mag[0][::mms_avg_fs], 'y': mms2_fcp_arr/4}) #Helium gyrofrequency
        pytplot.store_data('mms2_fcO', data={'x': mms2_mag[0][::mms_avg_fs], 'y': mms2_fcp_arr/16}) #Oxygen gyrofrequency
        pytplot.store_data('mms2_plotting_extracted', data=['mms2_dpwrspc_extracted', 'mms2_fcp', 'mms2_fcHe', 'mms2_fcO', 'mms2_imag_mask'])
        pytplot.options('mms2_imag_mask', 'spec', 1)
        pytplot.options('mms2_imag_mask', 'colormap', 'binary')
        pytplot.options('mms2_imag_mask', 'alpha', 0.67)
        pytplot.options('mms2_imag_mask', 'second_axis_size', 0.25)
        pytplot.options('mms2_imag_mask', 'zrange', [0,1])
        pytplot.options('mms2_dpwrspc_extracted', 'colormap', 'jet')
        pytplot.options('mms2_dpwrspc_extracted', 'data_gap', 10*60)
        pytplot.options('mms2_fcp', 'color', 'white')
        pytplot.options('mms2_fcp', 'thick', 1)
        pytplot.options('mms2_fcHe', 'color', 'white')
        pytplot.options('mms2_fcHe', 'thick', 1)
        pytplot.options('mms2_fcO', 'color', 'white')
        pytplot.options('mms2_fcO', 'thick', 1)
        pytplot.options('mms2_plotting_extracted', 'ytitle','MMS\n2\n')
        pytplot.options('mms2_plotting_extracted', 'ztitle',' ')
        pytplot.options('mms2_plotting_extracted', 'ylog', False)
        pytplot.options('mms2_dpwrspc_extracted', 'zlog', True)
        pytplot.options('mms2_imag_mask', 'zlog', False)
        pytplot.options('mms2_dpwrspc_extracted', 'second_axis_size', 0)
        pytplot.options('mms2_plotting_extracted', 'yrange', [0,4])
        pytplot.options('mms2_dpwrspc_extracted', 'zrange', [1e-1, 1e4])
        pytplot.options('mms2_dpwrspc_extracted', 'spec', 1)

        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms2-time_new', mms2_coord_new[0])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms2-coord_new', mms2_coord_new[1])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms2-l_new', mms2_l_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms2-imag', mms2_imag_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms2-spec_new', mms2_spec_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms2-spec_norm', mms2_spec/mms2_median)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms2-mask', mms2_mask)

        mms3_coord_new = time_align(mms3_coord[0], mms3_t, mms3_coord[1]/Re)
        mms3_l_new = time_align(mms3_l[0], mms3_t, mms3_l[1])
        mms3_imag_new = time_align(mms3_imag[0], mms3_t, mms3_imag[1])
        mms3_imag_new[1][np.where(mms3_imag_new[1] > 0)[0]] = 1
        mms3_imag_new[1] = magpause(mag_min_r, mms3_coord_new[1][:,0], mms3_coord_new[1][:,1], mms3_imag_new[1], mms3_l_new[1], mms_l_min)
        mms3_spec_new = np.zeros(mms3_spec.shape)*np.nan
        mms3_spec_new[np.where(mms3_imag_new[1] == 1)[0],:] = mms3_spec[np.where(mms3_imag_new[1] == 1)[0],:]
        mms3_median, mms3_std = median_extraction(mms3_t, mms3_freq, mms3_spec_new)
        mms3_mask = np.zeros(mms3_spec.shape)*np.nan
        mms3_mask[np.where(mms3_imag_new[1] == 0)[0],:] = 1
        pytplot.store_data('mms3_dpwrspc_extracted', data={'x': mms3_t, 'y': mms3_spec/mms3_median, 'v': mms3_freq})
        pytplot.store_data('mms3_imag_mask', data={'x': mms3_t, 'y': mms3_mask, 'v': mms3_freq})
        pytplot.store_data('mms3_fcp', data={'x': mms3_mag[0][::mms_avg_fs], 'y': mms3_fcp_arr}) #storing as a pytplot variable
        pytplot.store_data('mms3_fcHe', data={'x': mms3_mag[0][::mms_avg_fs], 'y': mms3_fcp_arr/4}) #Helium gyrofrequency
        pytplot.store_data('mms3_fcO', data={'x': mms3_mag[0][::mms_avg_fs], 'y': mms3_fcp_arr/16}) #Oxygen gyrofrequency
        pytplot.store_data('mms3_plotting_extracted', data=['mms3_dpwrspc_extracted', 'mms3_fcp', 'mms3_fcHe', 'mms3_fcO', 'mms3_imag_mask'])
        pytplot.options('mms3_imag_mask', 'spec', 1)
        pytplot.options('mms3_imag_mask', 'colormap', 'binary')
        pytplot.options('mms3_imag_mask', 'alpha', 0.67)
        pytplot.options('mms3_imag_mask', 'second_axis_size', 0.25)
        pytplot.options('mms3_imag_mask', 'zrange', [0,1])
        pytplot.options('mms3_dpwrspc_extracted', 'colormap', 'jet')
        pytplot.options('mms3_dpwrspc_extracted', 'data_gap', 10*60)
        pytplot.options('mms3_fcp', 'color', 'white')
        pytplot.options('mms3_fcp', 'thick', 1)
        pytplot.options('mms3_fcHe', 'color', 'white')
        pytplot.options('mms3_fcHe', 'thick', 1)
        pytplot.options('mms3_fcO', 'color', 'white')
        pytplot.options('mms3_fcO', 'thick', 1)
        pytplot.options('mms3_plotting_extracted', 'ytitle','MMS\n3\n')
        pytplot.options('mms3_plotting_extracted', 'ztitle',' ')
        pytplot.options('mms3_plotting_extracted', 'ylog', False)
        pytplot.options('mms3_dpwrspc_extracted', 'zlog', True)
        pytplot.options('mms3_imag_mask', 'zlog', False)
        pytplot.options('mms3_dpwrspc_extracted', 'second_axis_size', 0)
        pytplot.options('mms3_plotting_extracted', 'yrange', [0,4])
        pytplot.options('mms3_dpwrspc_extracted', 'zrange', [1e-1, 1e4])
        pytplot.options('mms3_dpwrspc_extracted', 'spec', 1)

        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms3-time_new', mms3_coord_new[0])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms3-coord_new', mms3_coord_new[1])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms3-l_new', mms3_l_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms3-imag', mms3_imag_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms3-spec_new', mms3_spec_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms3-spec_norm', mms3_spec/mms3_median)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms3-mask', mms3_mask)

        mms4_coord_new = time_align(mms4_coord[0], mms4_t, mms4_coord[1]/Re)
        mms4_l_new = time_align(mms4_l[0], mms4_t, mms4_l[1])
        mms4_imag_new = time_align(mms4_imag[0], mms4_t, mms4_imag[1])
        mms4_imag_new[1][np.where(mms4_imag_new[1] > 0)[0]] = 1
        mms4_imag_new[1] = magpause(mag_min_r, mms4_coord_new[1][:,0], mms4_coord_new[1][:,1], mms4_imag_new[1], mms4_l_new[1], mms_l_min)
        mms4_spec_new = np.zeros(mms4_spec.shape)*np.nan
        mms4_spec_new[np.where(mms4_imag_new[1] == 1)[0],:] = mms4_spec[np.where(mms4_imag_new[1] == 1)[0],:]
        mms4_median, mms4_std = median_extraction(mms4_t, mms4_freq, mms4_spec_new)
        mms4_mask = np.zeros(mms4_spec.shape)*np.nan
        mms4_mask[np.where(mms4_imag_new[1] == 0)[0],:] = 1
        pytplot.store_data('mms4_dpwrspc_extracted', data={'x': mms4_t, 'y': mms4_spec/mms4_median, 'v': mms4_freq})
        pytplot.store_data('mms4_imag_mask', data={'x': mms4_t, 'y': mms4_mask, 'v': mms4_freq})
        pytplot.store_data('mms4_fcp', data={'x': mms4_mag[0][::mms_avg_fs], 'y': mms4_fcp_arr}) #storing as a pytplot variable
        pytplot.store_data('mms4_fcHe', data={'x': mms4_mag[0][::mms_avg_fs], 'y': mms4_fcp_arr/4}) #Helium gyrofrequency
        pytplot.store_data('mms4_fcO', data={'x': mms4_mag[0][::mms_avg_fs], 'y': mms4_fcp_arr/16}) #Oxygen gyrofrequency
        pytplot.store_data('mms4_plotting_extracted', data=['mms4_dpwrspc_extracted', 'mms4_fcp', 'mms4_fcHe', 'mms4_fcO', 'mms4_imag_mask'])
        pytplot.options('mms4_imag_mask', 'spec', 1)
        pytplot.options('mms4_imag_mask', 'colormap', 'binary')
        pytplot.options('mms4_imag_mask', 'alpha', 0.67)
        pytplot.options('mms4_imag_mask', 'second_axis_size', 0.25)
        pytplot.options('mms4_imag_mask', 'zrange', [0,1])
        pytplot.options('mms4_dpwrspc_extracted', 'colormap', 'jet')
        pytplot.options('mms4_dpwrspc_extracted', 'data_gap', 10*60)
        pytplot.options('mms4_fcp', 'color', 'white')
        pytplot.options('mms4_fcp', 'thick', 1)
        pytplot.options('mms4_fcHe', 'color', 'white')
        pytplot.options('mms4_fcHe', 'thick', 1)
        pytplot.options('mms4_fcO', 'color', 'white')
        pytplot.options('mms4_fcO', 'thick', 1)
        pytplot.options('mms4_plotting_extracted', 'ytitle','MMS\n4\n           Frequency [Hz]')
        pytplot.options('mms4_plotting_extracted', 'ztitle','\n           Normalized Power Spectral Density')
        pytplot.options('mms4_plotting_extracted', 'ylog', False)
        pytplot.options('mms4_dpwrspc_extracted', 'zlog', True)
        pytplot.options('mms4_imag_mask', 'zlog', False)
        pytplot.options('mms4_dpwrspc_extracted', 'second_axis_size', 0)
        pytplot.options('mms4_plotting_extracted', 'yrange', [0,4])
        pytplot.options('mms4_dpwrspc_extracted', 'zrange', [1e-1, 1e4])
        pytplot.options('mms4_dpwrspc_extracted', 'spec', 1)

        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms4-time_new', mms4_coord_new[0])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms4-coord_new', mms4_coord_new[1])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms4-l_new', mms4_l_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms4-imag', mms4_imag_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms4-spec_new', mms4_spec_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms4-spec_norm', mms4_spec/mms4_median)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_mms4-mask', mms4_mask)

        # THEMIS
        th_l_min = 2.5 #minimum l-shell for valid measurements
        th_fs = 4
        th_avg_fs = th_fs*60 #bin size for averaging fcp array, 1-minute average
        
        tha_coord_new = time_align(tha_coord[0], tha_t, tha_coord[1])
        tha_l_new = time_align(tha_l[0], tha_t, tha_l[1])
        tha_imag = np.zeros(len(tha_dmagpause[1]), dtype=int)
        tha_imag_int = np.where(tha_dmagpause[1] < -1.5)[0] # identifies where the spacecraft is greater than 1.5Re from the magnetopause
        tha_imag[tha_imag_int] = 1 # value of 1 is where the spacecraft is in the inner magnetosphere (0 is outside)
        tha_imag_new = time_align(tha_dmagpause[0], tha_t, tha_imag)
        tha_imag_new[1] = magpause(mag_min_r, tha_coord_new[1][:,0], tha_coord_new[1][:,1], tha_imag_new[1], tha_l_new[1], th_l_min)
        tha_spec_new = np.zeros(tha_spec.shape)*np.nan
        tha_spec_new[np.where(tha_imag_new[1] == 1)[0],:] = tha_spec[np.where(tha_imag_new[1] == 1)[0],:]
        tha_median, tha_std = median_extraction(tha_t, tha_freq, tha_spec_new)
        tha_mask = np.zeros(tha_spec.shape)*np.nan
        tha_mask[np.where(tha_imag_new[1] == 0)[0],:] = 1
        pytplot.store_data('tha_dpwrspc_extracted', data={'x': tha_t, 'y': tha_spec/tha_median, 'v': tha_freq})
        pytplot.store_data('tha_imag_mask', data={'x': tha_t, 'y': tha_mask, 'v': tha_freq})
        pytplot.store_data('tha_fcp', data={'x': tha_mag[0][::th_avg_fs], 'y': tha_fcp_arr}) #storing as a pytplot variable
        pytplot.store_data('tha_fcHe', data={'x': tha_mag[0][::th_avg_fs], 'y': tha_fcp_arr/4}) #Helium gyrofrequency
        pytplot.store_data('tha_fcO', data={'x': tha_mag[0][::th_avg_fs], 'y': tha_fcp_arr/16}) #Oxygen gyrofrequency
        pytplot.store_data('tha_plotting_extracted', data=['tha_dpwrspc_extracted', 'tha_fcp', 'tha_fcHe', 'tha_fcO', 'tha_imag_mask'])
        pytplot.options('tha_imag_mask', 'spec', 1)
        pytplot.options('tha_imag_mask', 'colormap', 'binary')
        pytplot.options('tha_imag_mask', 'alpha', 0.67)
        pytplot.options('tha_imag_mask', 'second_axis_size', 0.25)
        pytplot.options('tha_imag_mask', 'zrange', [0,1])
        pytplot.options('tha_dpwrspc_extracted', 'colormap', 'jet')
        pytplot.options('tha_dpwrspc_extracted', 'data_gap', 10*60)
        pytplot.options('tha_fcp', 'color', 'white')
        pytplot.options('tha_fcp', 'thick', 1)
        pytplot.options('tha_fcHe', 'color', 'white')
        pytplot.options('tha_fcHe', 'thick', 1)
        pytplot.options('tha_fcO', 'color', 'white')
        pytplot.options('tha_fcO', 'thick', 1)
        pytplot.options('tha_plotting_extracted', 'charsize', 10)
        pytplot.options('tha_plotting_extracted', 'ytitle','THEMIS\nA\n\n')
        pytplot.options('tha_plotting_extracted', 'ztitle',' ')
        pytplot.options('tha_plotting_extracted', 'ylog', False)
        pytplot.options('tha_dpwrspc_extracted', 'zlog', True)
        pytplot.options('tha_imag_mask', 'zlog', False)
        pytplot.options('tha_dpwrspc_extracted', 'second_axis_size', 0)
        pytplot.options('tha_plotting_extracted', 'yrange', [0,2])
        pytplot.options('tha_dpwrspc_extracted', 'zrange', [1e-1, 1e4])
        pytplot.options('tha_dpwrspc_extracted', 'spec', 1)

        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_tha-time_new', tha_coord_new[0])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_tha-coord_new', tha_coord_new[1])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_tha-l_new', tha_l_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_tha-imag', tha_imag_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_tha-spec_new', tha_spec_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_tha-spec_norm', tha_spec/tha_median)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_tha-mask', tha_mask)

        thd_coord_new = time_align(thd_coord[0], thd_t, thd_coord[1])
        thd_l_new = time_align(thd_l[0], thd_t, thd_l[1])
        thd_imag = np.zeros(len(thd_dmagpause[1]), dtype=int)
        thd_imag_int = np.where(thd_dmagpause[1] < -1.5)[0] # identifies where the spacecraft is greater than 1.5Re from the magnetopause
        thd_imag[thd_imag_int] = 1 # value of 1 is where the spacecraft is in the inner magnetosphere (0 is outside)
        thd_imag_new = time_align(thd_dmagpause[0], thd_t, thd_imag)
        thd_imag_new[1] = magpause(mag_min_r, thd_coord_new[1][:,0], thd_coord_new[1][:,1], thd_imag_new[1], thd_l_new[1], th_l_min)
        thd_spec_new = np.zeros(thd_spec.shape)*np.nan
        thd_spec_new[np.where(thd_imag_new[1] == 1)[0],:] = thd_spec[np.where(thd_imag_new[1] == 1)[0],:]
        thd_median, thd_std = median_extraction(thd_t, thd_freq, thd_spec_new)
        thd_mask = np.zeros(thd_spec.shape)*np.nan
        thd_mask[np.where(thd_imag_new[1] == 0)[0],:] = 1
        pytplot.store_data('thd_dpwrspc_extracted', data={'x': thd_t, 'y': thd_spec/thd_median, 'v': thd_freq})
        pytplot.store_data('thd_imag_mask', data={'x': thd_t, 'y': thd_mask, 'v': thd_freq})
        pytplot.store_data('thd_fcp', data={'x': thd_mag[0][::th_avg_fs], 'y': thd_fcp_arr}) #storing as a pytplot variable
        pytplot.store_data('thd_fcHe', data={'x': thd_mag[0][::th_avg_fs], 'y': thd_fcp_arr/4}) #Helium gyrofrequency
        pytplot.store_data('thd_fcO', data={'x': thd_mag[0][::th_avg_fs], 'y': thd_fcp_arr/16}) #Oxygen gyrofrequency
        pytplot.store_data('thd_plotting_extracted', data=['thd_dpwrspc_extracted', 'thd_fcp', 'thd_fcHe', 'thd_fcO', 'thd_imag_mask'])
        pytplot.options('thd_imag_mask', 'spec', 1)
        pytplot.options('thd_imag_mask', 'colormap', 'binary')
        pytplot.options('thd_imag_mask', 'alpha', 0.67)
        pytplot.options('thd_imag_mask', 'second_axis_size', 0.25)
        pytplot.options('thd_imag_mask', 'zrange', [0,1])
        pytplot.options('thd_dpwrspc_extracted', 'colormap', 'jet')
        pytplot.options('thd_dpwrspc_extracted', 'data_gap', 10*60)
        pytplot.options('thd_fcp', 'color', 'white')
        pytplot.options('thd_fcp', 'thick', 1)
        pytplot.options('thd_fcHe', 'color', 'white')
        pytplot.options('thd_fcHe', 'thick', 1)
        pytplot.options('thd_fcO', 'color', 'white')
        pytplot.options('thd_fcO', 'thick', 1)
        pytplot.options('thd_plotting_extracted', 'charsize', 10)
        pytplot.options('thd_plotting_extracted', 'ytitle','THEMIS\nD\n\n')
        pytplot.options('thd_plotting_extracted', 'ztitle',' ')
        pytplot.options('thd_plotting_extracted', 'ylog', False)
        pytplot.options('thd_dpwrspc_extracted', 'zlog', True)
        pytplot.options('thd_imag_mask', 'zlog', False)
        pytplot.options('thd_dpwrspc_extracted', 'second_axis_size', 0)
        pytplot.options('thd_plotting_extracted', 'yrange', [0,2])
        pytplot.options('thd_dpwrspc_extracted', 'zrange', [1e-1, 1e4])
        pytplot.options('thd_dpwrspc_extracted', 'spec', 1)

        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_thd-time_new', thd_coord_new[0])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_thd-coord_new', thd_coord_new[1])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_thd-l_new', thd_l_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_thd-imag', thd_imag_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_thd-spec_new', thd_spec_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_thd-spec_norm', thd_spec/thd_median)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_thd-mask', thd_mask)

        the_coord_new = time_align(the_coord[0], the_t, the_coord[1])
        the_l_new = time_align(the_l[0], the_t, the_l[1])
        the_imag = np.zeros(len(the_dmagpause[1]), dtype=int)
        the_imag_int = np.where(the_dmagpause[1] < -1.5)[0] # identifies where the spacecraft is greater than 1.5Re from the magnetopause
        the_imag[the_imag_int] = 1 # value of 1 is where the spacecraft is in the inner magnetosphere (0 is outside)
        the_imag_new = time_align(the_dmagpause[0], the_t, the_imag)
        the_imag_new[1] = magpause(mag_min_r, the_coord_new[1][:,0], the_coord_new[1][:,1], the_imag_new[1], the_l_new[1], th_l_min)
        the_spec_new = np.zeros(the_spec.shape)*np.nan
        the_spec_new[np.where(the_imag_new[1] == 1)[0],:] = the_spec[np.where(the_imag_new[1] == 1)[0],:]
        the_median, the_std = median_extraction(the_t, the_freq, the_spec_new)
        the_mask = np.zeros(the_spec.shape)*np.nan
        the_mask[np.where(the_imag_new[1] == 0)[0],:] = 1
        pytplot.store_data('the_dpwrspc_extracted', data={'x': the_t, 'y': the_spec/the_median, 'v': the_freq})
        pytplot.store_data('the_imag_mask', data={'x': the_t, 'y': the_mask, 'v': the_freq})
        pytplot.store_data('the_fcp', data={'x': the_mag[0][::th_avg_fs], 'y': the_fcp_arr}) #storing as a pytplot variable
        pytplot.store_data('the_fcHe', data={'x': the_mag[0][::th_avg_fs], 'y': the_fcp_arr/4}) #Helium gyrofrequency
        pytplot.store_data('the_fcO', data={'x': the_mag[0][::th_avg_fs], 'y': the_fcp_arr/16}) #Oxygen gyrofrequency
        pytplot.store_data('the_plotting_extracted', data=['the_dpwrspc_extracted', 'the_fcp', 'the_fcHe', 'the_fcO', 'the_imag_mask'])
        pytplot.options('the_imag_mask', 'spec', 1)
        pytplot.options('the_imag_mask', 'colormap', 'binary')
        pytplot.options('the_imag_mask', 'alpha', 0.67)
        pytplot.options('the_imag_mask', 'second_axis_size', 0.25)
        pytplot.options('the_imag_mask', 'zrange', [0,1])
        pytplot.options('the_dpwrspc_extracted', 'colormap', 'jet')
        pytplot.options('the_dpwrspc_extracted', 'data_gap', 10*60)
        pytplot.options('the_fcp', 'color', 'white')
        pytplot.options('the_fcp', 'thick', 1)
        pytplot.options('the_fcHe', 'color', 'white')
        pytplot.options('the_fcHe', 'thick', 1)
        pytplot.options('the_fcO', 'color', 'white')
        pytplot.options('the_fcO', 'thick', 1)
        pytplot.options('the_plotting_extracted', 'charsize', 10)
        pytplot.options('the_plotting_extracted', 'ytitle','THEMIS\nE\n\n')
        pytplot.options('the_plotting_extracted', 'ztitle',' ')
        pytplot.options('the_plotting_extracted', 'ylog', False)
        pytplot.options('the_dpwrspc_extracted', 'zlog', True)
        pytplot.options('the_imag_mask', 'zlog', False)
        pytplot.options('the_dpwrspc_extracted', 'second_axis_size', 0)
        pytplot.options('the_plotting_extracted', 'yrange', [0,2])
        pytplot.options('the_dpwrspc_extracted', 'zrange', [1e-1, 1e4])
        pytplot.options('the_dpwrspc_extracted', 'spec', 1)

        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_the-time_new', the_coord_new[0])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_the-coord_new', the_coord_new[1])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_the-l_new', the_l_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_the-imag', the_imag_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_the-spec_new', the_spec_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_the-spec_norm', the_spec/the_median)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_the-mask', the_mask)

        # GOES
        goes_l_min = 2.5 #minimum l-shell for valid measurements
        goes_fs = 1/0.512
        goes_avg_fs = 2 * 60 #approximately 1-minute average (fs is 1/0.512 instead of 2)
        
        g15_coord_new = time_align(g15_coord[0], g15_t, g15_coord[1])
        g15_l_new = time_align(g15_l[0], g15_t, g15_l[1])
        g15_imag = np.zeros(len(g15_dmagpause[1]), dtype=int)
        g15_imag_int = np.where(g15_dmagpause[1] < -1.5)[0]
        g15_imag[g15_imag_int] = 1
        g15_imag_new = time_align(g15_dmagpause[0], g15_t, g15_imag)
        g15_imag_new[1] = magpause(mag_min_r, g15_coord_new[1][:,0], g15_coord_new[1][:,1], g15_imag_new[1], g15_l_new[1], goes_l_min)
        g15_spec_new = np.zeros(g15_spec.shape)*np.nan
        g15_spec_new[np.where(g15_imag_new[1] == 1)[0],:] = g15_spec[np.where(g15_imag_new[1] == 1)[0],:]
        g15_median, g15_std = median_extraction(g15_t, g15_freq, g15_spec_new)
        g15_mask = np.zeros(g15_spec.shape)*np.nan
        g15_mask[np.where(g15_imag_new[1] == 0)[0],:] = 1
        pytplot.store_data('g15_dpwrspc_extracted', data={'x': g15_t, 'y': g15_spec/g15_median, 'v': g15_freq})
        pytplot.store_data('g15_imag_mask', data={'x': g15_t, 'y': g15_mask, 'v': g15_freq})
        pytplot.store_data('g15_fcp', data={'x': g15_mag[0][::goes_avg_fs], 'y': g15_fcp_arr}) #storing as a pytplot variable
        pytplot.store_data('g15_fcHe', data={'x': g15_mag[0][::goes_avg_fs], 'y': g15_fcp_arr/4}) #Helium gyrofrequency
        pytplot.store_data('g15_fcO', data={'x': g15_mag[0][::goes_avg_fs], 'y': g15_fcp_arr/16}) #Oxygen gyrofrequency
        pytplot.store_data('g15_plotting_extracted', data=['g15_dpwrspc_extracted', 'g15_fcp', 'g15_fcHe', 'g15_fcO', 'g15_imag_mask'])
        pytplot.options('g15_imag_mask', 'spec', 1)
        pytplot.options('g15_imag_mask', 'colormap', 'binary')
        pytplot.options('g15_imag_mask', 'alpha', 0.67)
        pytplot.options('g15_imag_mask', 'second_axis_size', 0.25)
        pytplot.options('g15_imag_mask', 'zrange', [0,1])
        pytplot.options('g15_dpwrspc_extracted', 'colormap', 'jet')
        pytplot.options('g15_dpwrspc_extracted', 'data_gap', 10*60)
        pytplot.options('g15_fcp', 'color', 'white')
        pytplot.options('g15_fcp', 'thick', 1)
        pytplot.options('g15_fcHe', 'color', 'white')
        pytplot.options('g15_fcHe', 'thick', 1)
        pytplot.options('g15_fcO', 'color', 'white')
        pytplot.options('g15_fcO', 'thick', 1)
        pytplot.options('g15_plotting_extracted', 'ytitle','\nGOES\n15\n')
        pytplot.options('g15_plotting_extracted', 'ztitle',' ')
        pytplot.options('g15_plotting_extracted', 'ylog', False)
        pytplot.options('g15_dpwrspc_extracted', 'zlog', True)
        pytplot.options('g15_imag_mask', 'zlog', False)
        pytplot.options('g15_dpwrspc_extracted', 'second_axis_size', 0)
        pytplot.options('g15_plotting_extracted', 'yrange', [0,1])
        pytplot.options('g15_dpwrspc_extracted', 'zrange', [1e-1, 1e4])
        pytplot.options('g15_dpwrspc_extracted', 'spec', 1)

        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_g15-time_new', g15_coord_new[0])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_g15-coord_new', g15_coord_new[1])
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_g15-l_new', g15_l_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_g15-imag', g15_imag_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_g15-spec_new', g15_spec_new)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_g15-spec_norm', g15_spec/g15_median)
        np.save(file_path+'Satellite Storm Data/'+storm_day+'_'+storm_phase+'_g15-mask', g15_mask)

        pytplot.tplot_options('xmargin', [0.15,0.15]) #left/right margins, inches
        pytplot.tplot_options('axis_font_size', 9)
        pytplot.xlim(trange[0], trange[1])
        pytplot.tplot_options('title', storm_day+' '+storm_phase+' Phase')
        pytplot.tplot(['rbspa_plotting_extracted', 'rbspb_plotting_extracted', \
                            'mms1_plotting_extracted', 'mms2_plotting_extracted', 'mms3_plotting_extracted', 'mms4_plotting_extracted', \
                            'tha_plotting_extracted', 'thd_plotting_extracted', 'the_plotting_extracted', \
                                'g15_plotting_extracted'], vert_spacing=30, dpi=150, \
                                    save_png=file_path+storm_day+'_'+storm_phase+'_norm')

        # # %% EMIC Wave Identification

        # Van Allen Probes
        rbsp_f_max = 6.

        rbspa_fcp_new = time_align(rbspa_mag[0][::rbsp_avg_fs], rbspa_t, rbspa_fcp_arr)
        rbspa_t_mask = broadband_mask(rbspa_spec_new/rbspa_median, rbspa_freq, rbspa_t, rbspa_fcp_new[1], 'rbsp')
        rbspa_peaks_ind, rbspa_peaks, rbspa_peaks_times, rbspa_n_peaks = spec_peaks(rbspa_t, rbspa_spec_new/rbspa_median, rbspa_freq, rbsp_fs, rbspa_fcp_new[1]/16, rbspa_fcp_new[1], rbsp_f_max, rbspa_t_mask)#, rbspa_l_new[1], rbsp_l_min, rbspa_imag_new[1])

        rbspa_tind = np.where((rbspa_coord[0] >= trange[0]) & (rbspa_coord[0] <= trange[1]))[0]
        rbspa_mltind = np.where((rbspa_mlt[0] >= trange[0]) & (rbspa_mlt[0] <= trange[1]))[0]
        rbspa_peaks_tind = [[],[],[],[]]
        rbspa_peaks_mltind = [[],[],[],[]]
        for l in range(rbspa_n_peaks):
            if np.isnan(rbspa_peaks_times[0,l]): continue 
            elif storm_day == '2019-05-10': continue #removing instrument contamination in 2019-05-10 storm
            else:
                start = (np.abs(rbspa_coord[0] - rbspa_peaks_times[0,l])).argmin()
                end = (np.abs(rbspa_coord[0] - rbspa_peaks_times[1,l])).argmin()
                for ind in range(start,end+1): 
                    rbspa_peaks_tind[0].append(ind-rbspa_tind[0])
                    rbspa_peaks_tind[1].append(rbspa_coord[0][ind])
                    rbspa_peaks_tind[2].append(np.nan)
                    rbspa_peaks_tind[3].append(np.nanmean(rbspa_peaks[1][np.where(rbspa_peaks[2] == list(dict.fromkeys(rbspa_peaks[2]))[l])[0]]))
                mlt_start = (np.abs(rbspa_mlt[0] - rbspa_peaks_times[0,l])).argmin()
                mlt_end = (np.abs(rbspa_mlt[0] - rbspa_peaks_times[1,l])).argmin()
                for mlt_ind in range(mlt_start,mlt_end+1): 
                    rbspa_peaks_mltind[0].append(mlt_ind-rbspa_mltind[0])
                    rbspa_peaks_mltind[1].append(rbspa_mlt[0][mlt_ind])
                    rbspa_peaks_mltind[2].append(np.nan)
                    rbspa_peaks_mltind[3].append(np.nanmean(rbspa_peaks[1][np.where(rbspa_peaks[2] == list(dict.fromkeys(rbspa_peaks[2]))[l])[0]]))
        if len(rbspa_peaks_tind[0]) > 0:
            for m in range(len(rbspa_peaks[0])):
                rbspa_peaks_tind[2][(np.abs(rbspa_peaks[0][m] - rbspa_peaks_tind[1])).argmin()] = rbspa_peaks[1][m]
                rbspa_peaks_mltind[2][(np.abs(rbspa_peaks[0][m] - rbspa_peaks_mltind[1])).argmin()] = rbspa_peaks[1][m]
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspa-peaks', rbspa_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspa-peaks-times', rbspa_peaks_times)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspa-n-peaks', rbspa_n_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspa', rbspa_peaks_tind)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspa-time-coord', rbspa_coord[0][rbspa_tind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspa-coord', rbspa_coord[1][rbspa_tind,:]/Re)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspa-time', rbspa_mlt[0][rbspa_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspa-mltind', rbspa_peaks_mltind)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspa-l', rbspa_l[1][rbspa_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspa-mlt', rbspa_mlt[1][rbspa_mltind])
        rbspa_imag_mltind = time_align(rbspa_imag_new[0], rbspa_mlt[0], rbspa_imag_new[1])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspa-imag', rbspa_imag_mltind[1][rbspa_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspa-fcp', rbspa_fcp_arr[rbspa_mltind])

        rbspb_fcp_new = time_align(rbspb_mag[0][::rbsp_avg_fs], rbspb_t, rbspb_fcp_arr)
        rbspb_t_mask = broadband_mask(rbspb_spec_new/rbspb_median, rbspb_freq, rbspb_t, rbspb_fcp_new[1], 'rbsp')
        rbspb_peaks_ind, rbspb_peaks, rbspb_peaks_times, rbspb_n_peaks = spec_peaks(rbspb_t, rbspb_spec_new/rbspb_median, rbspb_freq, rbsp_fs, rbspb_fcp_new[1]/16, rbspb_fcp_new[1], rbsp_f_max, rbspb_t_mask)#, rbspb_l_new[1], rbsp_l_min, rbspb_imag_new[1])

        rbspb_tind = np.where((rbspb_coord[0] >= trange[0]) & (rbspb_coord[0] <= trange[1]))[0]
        rbspb_mltind = np.where((rbspb_mlt[0] >= trange[0]) & (rbspb_mlt[0] <= trange[1]))[0]
        rbspb_peaks_tind = [[],[],[],[]]
        rbspb_peaks_mltind = [[],[],[],[]]
        for l in range(rbspb_n_peaks):
            if np.isnan(rbspb_peaks_times[0,l]): continue 
            else:
                start = (np.abs(rbspb_coord[0] - rbspb_peaks_times[0,l])).argmin()
                end = (np.abs(rbspb_coord[0] - rbspb_peaks_times[1,l])).argmin()
                for ind in range(start,end+1): 
                    rbspb_peaks_tind[0].append(ind-rbspb_tind[0])
                    rbspb_peaks_tind[1].append(rbspb_coord[0][ind])
                    rbspb_peaks_tind[2].append(np.nan)
                    rbspb_peaks_tind[3].append(np.nanmean(rbspb_peaks[1][np.where(rbspb_peaks[2] == list(dict.fromkeys(rbspb_peaks[2]))[l])[0]]))
                mlt_start = (np.abs(rbspb_mlt[0] - rbspb_peaks_times[0,l])).argmin()
                mlt_end = (np.abs(rbspb_mlt[0] - rbspb_peaks_times[1,l])).argmin()
                for mlt_ind in range(mlt_start,mlt_end+1): 
                    rbspb_peaks_mltind[0].append(mlt_ind-rbspb_mltind[0])
                    rbspb_peaks_mltind[1].append(rbspb_mlt[0][mlt_ind])
                    rbspb_peaks_mltind[2].append(np.nan)
                    rbspb_peaks_mltind[3].append(np.nanmean(rbspb_peaks[1][np.where(rbspb_peaks[2] == list(dict.fromkeys(rbspb_peaks[2]))[l])[0]]))
        if len(rbspb_peaks_tind[0]) > 0:
            for m in range(len(rbspb_peaks[0])):
                rbspb_peaks_tind[2][(np.abs(rbspb_peaks[0][m] - rbspb_peaks_tind[1])).argmin()] = rbspb_peaks[1][m]
                rbspb_peaks_mltind[2][(np.abs(rbspb_peaks[0][m] - rbspb_peaks_mltind[1])).argmin()] = rbspb_peaks[1][m]
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspb-peaks', rbspb_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspb-peaks-times', rbspb_peaks_times)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspb-n-peaks', rbspb_n_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspb', rbspb_peaks_tind)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspb-time-coord', rbspb_coord[0][rbspb_tind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspb-coord', rbspb_coord[1][rbspb_tind,:]/Re)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspb-time', rbspb_mlt[0][rbspb_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspb-mltind', rbspb_peaks_mltind)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspb-l', rbspb_l[1][rbspb_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspb-mlt', rbspb_mlt[1][rbspb_mltind])
        rbspb_imag_mltind = time_align(rbspb_imag_new[0], rbspb_mlt[0], rbspb_imag_new[1])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspb-imag', rbspb_imag_mltind[1][rbspb_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_rbspb-fcp', rbspb_fcp_arr[rbspb_mltind])

        # MMS
        mms_f_max = 4.
        
        mms1_fcp_new = time_align(mms1_mag[0][::mms_avg_fs], mms1_t, mms1_fcp_arr)
        mms1_t_mask = broadband_mask(mms1_spec_new/mms1_median, mms1_freq, mms1_t, mms1_fcp_new[1], 'mms')
        mms1_peaks_ind, mms1_peaks, mms1_peaks_times, mms1_n_peaks = spec_peaks(mms1_t, mms1_spec_new/mms1_median, mms1_freq, mms_fs, mms1_fcp_new[1]/16, mms1_fcp_new[1], mms_f_max, mms1_t_mask)#, mms1_l_new[1], mms_l_min, mms1_imag_new[1])
        
        mms1_l_mltind = time_align(mms1_l[0], rbspa_mlt[0], mms1_l[1])
        mms1_mlt_new = time_align(mms1_mlt[0], rbspa_mlt[0], mms1_mlt[1])
        mms1_fcp_min = time_align(mms1_mag[0][::mms_avg_fs], rbspa_mlt[0], mms1_fcp_arr) #convert it to 1-minute average; previously 1-minute in slow-survey data and 30-second in fast-survey data

        mms1_tind = np.where((mms1_coord[0] >= trange[0]) & (mms1_coord[0] <= trange[1]))[0]
        mms1_mltind = np.where((mms1_mlt_new[0] >= trange[0]) & (mms1_mlt_new[0] <= trange[1]))[0]
        mms1_peaks_tind = [[],[],[],[]]
        mms1_peaks_mltind = [[],[],[],[]]
        for l in range(mms1_n_peaks):
            if np.isnan(mms1_peaks_times[0,l]): continue 
            elif (mms1_peaks_times[0,l] > pyspedas.time_double('2015-11-06 20:30')) and (mms1_peaks_times[0,l] < pyspedas.time_double('2015-11-06 23:00')): continue #removing instrument contamination
            elif (mms1_peaks_times[0,l] > pyspedas.time_double('2016-08-02 13:00')) and (mms1_peaks_times[0,l] < pyspedas.time_double('2016-08-02 16:00')): continue
            elif (mms1_peaks_times[0,l] > pyspedas.time_double('2016-07-24 14:15')) and (mms1_peaks_times[0,l] < pyspedas.time_double('2016-07-24 16:45')): continue
            elif (mms1_peaks_times[0,l] > pyspedas.time_double('2016-10-13 05:00')) and (mms1_peaks_times[0,l] < pyspedas.time_double('2016-10-13 06:30')): continue
            elif (mms1_peaks_times[0,l] > pyspedas.time_double('2019-05-10 21:15')) and (mms1_peaks_times[0,l] < pyspedas.time_double('2019-05-10 22:00')): continue
            else:
                start = (np.abs(mms1_coord[0] - mms1_peaks_times[0,l])).argmin()
                end = (np.abs(mms1_coord[0] - mms1_peaks_times[1,l])).argmin()
                for ind in range(start,end+1): 
                    mms1_peaks_tind[0].append(ind-mms1_tind[0])
                    mms1_peaks_tind[1].append(mms1_coord[0][ind])
                    mms1_peaks_tind[2].append(np.nan)
                    mms1_peaks_tind[3].append(np.nanmean(mms1_peaks[1][np.where(mms1_peaks[2] == list(dict.fromkeys(mms1_peaks[2]))[l])[0]]))
                mlt_start = (np.abs(mms1_mlt_new[0] - mms1_peaks_times[0,l])).argmin()
                mlt_end = (np.abs(mms1_mlt_new[0] - mms1_peaks_times[1,l])).argmin()
                for mlt_ind in range(mlt_start,mlt_end+1): 
                    mms1_peaks_mltind[0].append(mlt_ind-mms1_mltind[0])
                    mms1_peaks_mltind[1].append(mms1_mlt[0][mlt_ind])
                    mms1_peaks_mltind[2].append(np.nan)
                    mms1_peaks_mltind[3].append(np.nanmean(mms1_peaks[1][np.where(mms1_peaks[2] == list(dict.fromkeys(mms1_peaks[2]))[l])[0]]))
        if len(mms1_peaks_tind[0]) > 0:
            for m in range(len(mms1_peaks[0])):
                mms1_peaks_tind[2][(np.abs(mms1_peaks[0][m] - mms1_peaks_tind[1])).argmin()] = mms1_peaks[1][m]
                mms1_peaks_mltind[2][(np.abs(mms1_peaks[0][m] - mms1_peaks_mltind[1])).argmin()] = mms1_peaks[1][m]
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms1-peaks', mms1_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms1-peaks-times', mms1_peaks_times)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms1-n-peaks', mms1_n_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms1', mms1_peaks_tind)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms1-time-coord', mms1_coord[0][mms1_tind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms1-coord', mms1_coord[1][mms1_tind,:]/Re)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms1-time', mms1_mlt_new[0][mms1_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms1-mltind', mms1_peaks_mltind)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms1-l', mms1_l_mltind[1][mms1_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms1-mlt', mms1_mlt_new[1][mms1_mltind])
        mms1_imag_mltind = time_align(mms1_imag_new[0], mms1_mlt_new[0], mms1_imag_new[1])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms1-imag', mms1_imag_mltind[1][mms1_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms1-fcp', mms1_fcp_min[1][mms1_mltind])

        mms2_fcp_new = time_align(mms2_mag[0][::mms_avg_fs], mms2_t, mms2_fcp_arr)
        mms2_t_mask = broadband_mask(mms2_spec_new/mms2_median, mms2_freq, mms2_t, mms2_fcp_new[1], 'mms')
        mms2_peaks_ind, mms2_peaks, mms2_peaks_times, mms2_n_peaks = spec_peaks(mms2_t, mms2_spec_new/mms2_median, mms2_freq, mms_fs, mms2_fcp_new[1]/16, mms2_fcp_new[1], mms_f_max, mms2_t_mask)#, mms2_l_new[1], mms_l_min, mms2_imag_new[1])
        
        mms2_l_mltind = time_align(mms2_l[0], rbspa_mlt[0], mms2_l[1])
        mms2_mlt_new = time_align(mms2_mlt[0], rbspa_mlt[0], mms2_mlt[1])
        mms2_fcp_min = time_align(mms2_mag[0][::mms_avg_fs], rbspa_mlt[0], mms2_fcp_arr)

        mms2_tind = np.where((mms2_coord[0] >= trange[0]) & (mms2_coord[0] <= trange[1]))[0]
        mms2_mltind = np.where((mms2_mlt_new[0] >= trange[0]) & (mms2_mlt_new[0] <= trange[1]))[0]
        mms2_peaks_tind = [[],[],[],[]]
        mms2_peaks_mltind = [[],[],[],[]]
        for l in range(mms2_n_peaks):
            if np.isnan(mms2_peaks_times[0,l]): continue 
            elif (mms2_peaks_times[0,l] > pyspedas.time_double('2015-11-06 20:30')) and (mms2_peaks_times[0,l] < pyspedas.time_double('2015-11-06 23:00')): continue #removing instrument contamination
            elif (mms2_peaks_times[0,l] > pyspedas.time_double('2016-08-02 13:00')) and (mms2_peaks_times[0,l] < pyspedas.time_double('2016-08-02 16:00')): continue
            elif (mms2_peaks_times[0,l] > pyspedas.time_double('2016-07-24 14:15')) and (mms2_peaks_times[0,l] < pyspedas.time_double('2016-07-24 16:45')): continue
            elif (mms2_peaks_times[0,l] > pyspedas.time_double('2016-10-13 05:00')) and (mms2_peaks_times[0,l] < pyspedas.time_double('2016-10-13 06:30')): continue
            elif (mms2_peaks_times[0,l] > pyspedas.time_double('2019-05-10 21:15')) and (mms2_peaks_times[0,l] < pyspedas.time_double('2019-05-10 22:00')): continue
            else:
                start = (np.abs(mms2_coord[0] - mms2_peaks_times[0,l])).argmin()
                end = (np.abs(mms2_coord[0] - mms2_peaks_times[1,l])).argmin()
                for ind in range(start,end+1): 
                    mms2_peaks_tind[0].append(ind-mms2_tind[0])
                    mms2_peaks_tind[1].append(mms2_coord[0][ind])
                    mms2_peaks_tind[2].append(np.nan)
                    mms2_peaks_tind[3].append(np.nanmean(mms2_peaks[1][np.where(mms2_peaks[2] == list(dict.fromkeys(mms2_peaks[2]))[l])[0]]))
                mlt_start = (np.abs(mms2_mlt_new[0] - mms2_peaks_times[0,l])).argmin()
                mlt_end = (np.abs(mms2_mlt_new[0] - mms2_peaks_times[1,l])).argmin()
                for mlt_ind in range(mlt_start,mlt_end+1): 
                    mms2_peaks_mltind[0].append(mlt_ind-mms2_mltind[0])
                    mms2_peaks_mltind[1].append(mms2_mlt[0][mlt_ind])
                    mms2_peaks_mltind[2].append(np.nan)
                    mms2_peaks_mltind[3].append(np.nanmean(mms2_peaks[1][np.where(mms2_peaks[2] == list(dict.fromkeys(mms2_peaks[2]))[l])[0]]))
        if len(mms2_peaks_tind[0]) > 0:
            for m in range(len(mms2_peaks[0])):
                mms2_peaks_tind[2][(np.abs(mms2_peaks[0][m] - mms2_peaks_tind[1])).argmin()] = mms2_peaks[1][m]
                mms2_peaks_mltind[2][(np.abs(mms2_peaks[0][m] - mms2_peaks_mltind[1])).argmin()] = mms2_peaks[1][m]
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms2-peaks', mms2_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms2-peaks-times', mms2_peaks_times)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms2-n-peaks', mms2_n_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms2', mms2_peaks_tind)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms2-time-coord', mms2_coord[0][mms2_tind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms2-coord', mms2_coord[1][mms2_tind,:]/Re)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms2-time', mms2_mlt_new[0][mms2_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms2-mltind', mms2_peaks_mltind)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms2-l', mms2_l_mltind[1][mms2_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms2-mlt', mms2_mlt_new[1][mms2_mltind])
        mms2_imag_mltind = time_align(mms2_imag_new[0], mms2_mlt_new[0], mms2_imag_new[1])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms2-imag', mms2_imag_mltind[1][mms2_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms2-fcp', mms2_fcp_min[1][mms2_mltind])

        mms3_fcp_new = time_align(mms3_mag[0][::mms_avg_fs], mms3_t, mms3_fcp_arr)
        mms3_t_mask = broadband_mask(mms3_spec_new/mms3_median, mms3_freq, mms3_t, mms3_fcp_new[1], 'mms')
        mms3_peaks_ind, mms3_peaks, mms3_peaks_times, mms3_n_peaks = spec_peaks(mms3_t, mms3_spec_new/mms3_median, mms3_freq, mms_fs, mms3_fcp_new[1]/16, mms3_fcp_new[1], mms_f_max, mms3_t_mask)#, mms3_l_new[1], mms_l_min, mms3_imag_new[1])

        mms3_l_mltind = time_align(mms3_l[0], rbspa_mlt[0], mms3_l[1])
        mms3_mlt_new = time_align(mms3_mlt[0], rbspa_mlt[0], mms3_mlt[1])
        mms3_fcp_min = time_align(mms3_mag[0][::mms_avg_fs], rbspa_mlt[0], mms3_fcp_arr)

        mms3_tind = np.where((mms3_coord[0] >= trange[0]) & (mms3_coord[0] <= trange[1]))[0]
        mms3_mltind = np.where((mms3_mlt_new[0] >= trange[0]) & (mms3_mlt_new[0] <= trange[1]))[0]
        mms3_peaks_tind = [[],[],[],[]]
        mms3_peaks_mltind = [[],[],[],[]]
        for l in range(mms3_n_peaks):
            if np.isnan(mms3_peaks_times[0,l]): continue 
            elif (mms3_peaks_times[0,l] > pyspedas.time_double('2015-11-06 20:30')) and (mms3_peaks_times[0,l] < pyspedas.time_double('2015-11-06 23:00')): continue #removing instrument contamination
            elif (mms3_peaks_times[0,l] > pyspedas.time_double('2016-08-02 13:00')) and (mms3_peaks_times[0,l] < pyspedas.time_double('2016-08-02 16:00')): continue
            elif (mms3_peaks_times[0,l] > pyspedas.time_double('2016-07-24 14:15')) and (mms3_peaks_times[0,l] < pyspedas.time_double('2016-07-24 16:45')): continue
            elif (mms3_peaks_times[0,l] > pyspedas.time_double('2016-10-13 05:00')) and (mms3_peaks_times[0,l] < pyspedas.time_double('2016-10-13 06:30')): continue
            elif (mms3_peaks_times[0,l] > pyspedas.time_double('2019-05-10 21:15')) and (mms3_peaks_times[0,l] < pyspedas.time_double('2019-05-10 22:00')): continue
            else:
                start = (np.abs(mms3_coord[0] - mms3_peaks_times[0,l])).argmin()
                end = (np.abs(mms3_coord[0] - mms3_peaks_times[1,l])).argmin()
                for ind in range(start,end+1): 
                    mms3_peaks_tind[0].append(ind-mms3_tind[0])
                    mms3_peaks_tind[1].append(mms3_coord[0][ind])
                    mms3_peaks_tind[2].append(np.nan)
                    mms3_peaks_tind[3].append(np.nanmean(mms3_peaks[1][np.where(mms3_peaks[2] == list(dict.fromkeys(mms3_peaks[2]))[l])[0]]))
                mlt_start = (np.abs(mms3_mlt_new[0] - mms3_peaks_times[0,l])).argmin()
                mlt_end = (np.abs(mms3_mlt_new[0] - mms3_peaks_times[1,l])).argmin()
                for mlt_ind in range(mlt_start,mlt_end+1): 
                    mms3_peaks_mltind[0].append(mlt_ind-mms3_mltind[0])
                    mms3_peaks_mltind[1].append(mms3_mlt[0][mlt_ind])
                    mms3_peaks_mltind[2].append(np.nan)
                    mms3_peaks_mltind[3].append(np.nanmean(mms3_peaks[1][np.where(mms3_peaks[2] == list(dict.fromkeys(mms3_peaks[2]))[l])[0]]))
        if len(mms3_peaks_tind[0]) > 0:
            for m in range(len(mms3_peaks[0])):
                mms3_peaks_tind[2][(np.abs(mms3_peaks[0][m] - mms3_peaks_tind[1])).argmin()] = mms3_peaks[1][m]
                mms3_peaks_mltind[2][(np.abs(mms3_peaks[0][m] - mms3_peaks_mltind[1])).argmin()] = mms3_peaks[1][m]
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms3-peaks', mms3_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms3-peaks-times', mms3_peaks_times)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms3-n-peaks', mms3_n_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms3', mms3_peaks_tind)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms3-time-coord', mms3_coord[0][mms3_tind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms3-coord', mms3_coord[1][mms3_tind,:]/Re)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms3-time', mms3_mlt_new[0][mms3_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms3-mltind', mms3_peaks_mltind)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms3-l', mms3_l_mltind[1][mms3_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms3-mlt', mms3_mlt_new[1][mms3_mltind])
        mms3_imag_mltind = time_align(mms3_imag_new[0], mms3_mlt_new[0], mms3_imag_new[1])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms3-imag', mms3_imag_mltind[1][mms3_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms3-fcp', mms3_fcp_min[1][mms3_mltind])

        mms4_fcp_new = time_align(mms4_mag[0][::mms_avg_fs], mms4_t, mms4_fcp_arr)
        mms4_t_mask = broadband_mask(mms4_spec_new/mms4_median, mms4_freq, mms4_t, mms4_fcp_new[1], 'mms')
        mms4_peaks_ind, mms4_peaks, mms4_peaks_times, mms4_n_peaks = spec_peaks(mms4_t, mms4_spec_new/mms4_median, mms4_freq, mms_fs, mms4_fcp_new[1]/16, mms4_fcp_new[1], mms_f_max, mms4_t_mask)#, mms4_l_new[1], mms_l_min, mms4_imag_new[1])
        
        mms4_l_mltind = time_align(mms4_l[0], rbspa_mlt[0], mms4_l[1])
        mms4_mlt_new = time_align(mms4_mlt[0], rbspa_mlt[0], mms4_mlt[1])
        mms4_fcp_min = time_align(mms4_mag[0][::mms_avg_fs], rbspa_mlt[0], mms4_fcp_arr)

        mms4_tind = np.where((mms4_coord[0] >= trange[0]) & (mms4_coord[0] <= trange[1]))[0]
        mms4_mltind = np.where((mms4_mlt_new[0] >= trange[0]) & (mms4_mlt_new[0] <= trange[1]))[0]
        mms4_peaks_tind = [[],[],[],[]]
        mms4_peaks_mltind = [[],[],[],[]]
        for l in range(mms4_n_peaks):
            if np.isnan(mms4_peaks_times[0,l]): continue 
            elif (mms4_peaks_times[0,l] > pyspedas.time_double('2015-11-06 20:30')) and (mms4_peaks_times[0,l] < pyspedas.time_double('2015-11-06 23:00')): continue #removing instrument contamination
            elif (mms4_peaks_times[0,l] > pyspedas.time_double('2016-08-02 13:00')) and (mms4_peaks_times[0,l] < pyspedas.time_double('2016-08-02 16:00')): continue
            elif (mms4_peaks_times[0,l] > pyspedas.time_double('2016-07-24 14:15')) and (mms4_peaks_times[0,l] < pyspedas.time_double('2016-07-24 16:45')): continue
            elif (mms4_peaks_times[0,l] > pyspedas.time_double('2016-10-13 05:00')) and (mms4_peaks_times[0,l] < pyspedas.time_double('2016-10-13 06:30')): continue
            elif (mms4_peaks_times[0,l] > pyspedas.time_double('2019-05-10 21:15')) and (mms4_peaks_times[0,l] < pyspedas.time_double('2019-05-10 22:00')): continue
            else:
                start = (np.abs(mms4_coord[0] - mms4_peaks_times[0,l])).argmin()
                end = (np.abs(mms4_coord[0] - mms4_peaks_times[1,l])).argmin()
                for ind in range(start,end+1): 
                    mms4_peaks_tind[0].append(ind-mms4_tind[0])
                    mms4_peaks_tind[1].append(mms4_coord[0][ind])
                    mms4_peaks_tind[2].append(np.nan)
                    mms4_peaks_tind[3].append(np.nanmean(mms4_peaks[1][np.where(mms4_peaks[2] == list(dict.fromkeys(mms4_peaks[2]))[l])[0]]))
                mlt_start = (np.abs(mms4_mlt_new[0] - mms4_peaks_times[0,l])).argmin()
                mlt_end = (np.abs(mms4_mlt_new[0] - mms4_peaks_times[1,l])).argmin()
                for mlt_ind in range(mlt_start,mlt_end+1): 
                    mms4_peaks_mltind[0].append(mlt_ind-mms4_mltind[0])
                    mms4_peaks_mltind[1].append(mms4_mlt[0][mlt_ind])
                    mms4_peaks_mltind[2].append(np.nan)
                    mms4_peaks_mltind[3].append(np.nanmean(mms4_peaks[1][np.where(mms4_peaks[2] == list(dict.fromkeys(mms4_peaks[2]))[l])[0]]))
        if len(mms4_peaks_tind[0]) > 0:
            for m in range(len(mms4_peaks[0])):
                mms4_peaks_tind[2][(np.abs(mms4_peaks[0][m] - mms4_peaks_tind[1])).argmin()] = mms4_peaks[1][m]
                mms4_peaks_mltind[2][(np.abs(mms4_peaks[0][m] - mms4_peaks_mltind[1])).argmin()] = mms4_peaks[1][m]
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms4-peaks', mms4_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms4-peaks-times', mms4_peaks_times)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms4-n-peaks', mms4_n_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms4', mms4_peaks_tind)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms4-time-coord', mms4_coord[0][mms4_tind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms4-coord', mms4_coord[1][mms4_tind,:]/Re)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms4-time', mms4_mlt_new[0][mms4_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms4-mltind', mms4_peaks_mltind)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms4-l', mms4_l_mltind[1][mms4_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms4-mlt', mms4_mlt_new[1][mms4_mltind])
        mms4_imag_mltind = time_align(mms4_imag_new[0], mms4_mlt_new[0], mms4_imag_new[1])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms4-imag', mms4_imag_mltind[1][mms4_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_mms4-fcp', mms4_fcp_min[1][mms4_mltind])

        # THEMIS
        th_f_max = 2.

        tha_fcp_new = time_align(tha_mag[0][::th_avg_fs], tha_t, tha_fcp_arr)
        tha_t_mask = broadband_mask(tha_spec_new/tha_median, tha_freq, tha_t, tha_fcp_new[1], 'th')
        tha_peaks_ind, tha_peaks, tha_peaks_times, tha_n_peaks = spec_peaks(tha_t, tha_spec_new/tha_median, tha_freq, th_fs, tha_fcp_new[1]/16, tha_fcp_new[1], th_f_max, tha_t_mask)#, tha_l_new[1], th_l_min, tha_imag_new[1])

        tha_fcp_min = np.zeros((2,len(tha_coord[0])))*np.nan
        tha_fcp_min[0] = tha_coord[0]
        for t in range(len(tha_coord[0])):
            if np.any((tha_mag[0][::th_avg_fs] >= tha_coord[0][t]) & (tha_mag[0][::th_avg_fs] < tha_coord[0][t]+60)):
                tha_fcp_min[1][t] = np.nanmean(tha_fcp_arr[np.where((tha_mag[0][::th_avg_fs] >= tha_coord[0][t]) & (tha_mag[0][::th_avg_fs] < tha_coord[0][t+1]))[0]])

        tha_tind = np.where((tha_coord[0] >= trange[0]) & (tha_coord[0] <= trange[1]))[0]
        tha_peaks_tind = [[],[],[],[]]
        for l in range(tha_n_peaks):
            if np.isnan(tha_peaks_times[0,l]): continue 
            elif (tha_peaks_times[0,l] > pyspedas.time_double('2016-10-13 03:30')) and (tha_peaks_times[0,l] < pyspedas.time_double('2016-10-13 04:00')): continue #removing instrument contamination
            else:
                start = (np.abs(tha_coord[0] - tha_peaks_times[0,l])).argmin()
                end = (np.abs(tha_coord[0] - tha_peaks_times[1,l])).argmin()
                for ind in range(start,end+1): 
                    tha_peaks_tind[0].append(ind-tha_tind[0])
                    tha_peaks_tind[1].append(tha_coord[0][ind])
                    tha_peaks_tind[2].append(np.nan)
                    tha_peaks_tind[3].append(np.nanmean(tha_peaks[1][np.where(tha_peaks[2] == list(dict.fromkeys(tha_peaks[2]))[l])[0]]))
        if len(tha_peaks_tind[0]) > 0:
            for m in range(len(tha_peaks[0])):
                tha_peaks_tind[2][(np.abs(tha_peaks[0][m] - tha_peaks_tind[1])).argmin()] = tha_peaks[1][m]
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_tha-peaks', tha_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_tha-peaks-times', tha_peaks_times)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_tha-n-peaks', tha_n_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_tha', tha_peaks_tind)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_tha-time', tha_coord[0][tha_tind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_tha-coord', tha_coord[1][tha_tind,:])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_tha-l', tha_l[1][tha_tind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_tha-mlt', tha_mlt[1][tha_tind])
        tha_imag_mltind = time_align(tha_imag_new[0], tha_coord[0], tha_imag_new[1])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_tha-imag', tha_imag_mltind[1][tha_tind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_tha-fcp', tha_fcp_min[1][tha_tind])

        thd_fcp_new = time_align(thd_mag[0][::th_avg_fs], thd_t, thd_fcp_arr)
        thd_t_mask = broadband_mask(thd_spec_new/thd_median, thd_freq, thd_t, thd_fcp_new[1], 'th')
        thd_peaks_ind, thd_peaks, thd_peaks_times, thd_n_peaks = spec_peaks(thd_t, thd_spec_new/thd_median, thd_freq, th_fs, thd_fcp_new[1]/16, thd_fcp_new[1], th_f_max, thd_t_mask)#, thd_l_new[1], th_l_min, thd_imag_new[1])

        thd_fcp_min = np.zeros((2,len(thd_coord[0])))*np.nan
        thd_fcp_min[0] = thd_coord[0]
        for t in range(len(thd_coord[0])):
            if np.any((thd_mag[0][::th_avg_fs] >= thd_coord[0][t]) & (thd_mag[0][::th_avg_fs] < thd_coord[0][t]+60)):
                thd_fcp_min[1][t] = np.nanmean(thd_fcp_arr[np.where((thd_mag[0][::th_avg_fs] >= thd_coord[0][t]) & (thd_mag[0][::th_avg_fs] < thd_coord[0][t+1]))[0]])

        thd_tind = np.where((thd_coord[0] >= trange[0]) & (thd_coord[0] <= trange[1]))[0]
        thd_peaks_tind = [[],[],[],[]]
        for l in range(thd_n_peaks):
            if np.isnan(thd_peaks_times[0,l]): continue 
            else:
                start = (np.abs(thd_coord[0] - thd_peaks_times[0,l])).argmin()
                end = (np.abs(thd_coord[0] - thd_peaks_times[1,l])).argmin()
                for ind in range(start,end+1): 
                    thd_peaks_tind[0].append(ind-thd_tind[0])
                    thd_peaks_tind[1].append(thd_coord[0][ind])
                    thd_peaks_tind[2].append(np.nan)
                    thd_peaks_tind[3].append(np.nanmean(thd_peaks[1][np.where(thd_peaks[2] == list(dict.fromkeys(thd_peaks[2]))[l])[0]]))
        if len(thd_peaks_tind[0]) > 0:
            for m in range(len(thd_peaks[0])):
                thd_peaks_tind[2][(np.abs(thd_peaks[0][m] - thd_peaks_tind[1])).argmin()] = thd_peaks[1][m]
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_thd-peaks', thd_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_thd-peaks-times', thd_peaks_times)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_thd-n-peaks', thd_n_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_thd', thd_peaks_tind)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_thd-time', thd_coord[0][thd_tind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_thd-coord', thd_coord[1][thd_tind,:])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_thd-l', thd_l[1][thd_tind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_thd-mlt', thd_mlt[1][thd_tind])
        thd_imag_mltind = time_align(thd_imag_new[0], thd_coord[0], thd_imag_new[1])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_thd-imag', thd_imag_mltind[1][thd_tind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_thd-fcp', thd_fcp_min[1][thd_tind])

        the_fcp_new = time_align(the_mag[0][::th_avg_fs], the_t, the_fcp_arr)
        the_t_mask = broadband_mask(the_spec_new/the_median, the_freq, the_t, the_fcp_new[1], 'th')
        the_peaks_ind, the_peaks, the_peaks_times, the_n_peaks = spec_peaks(the_t, the_spec_new/the_median, the_freq, th_fs, the_fcp_new[1]/16, the_fcp_new[1], th_f_max, the_t_mask)#, the_l_new[1], th_l_min, the_imag_new[1])

        the_fcp_min = np.zeros((2,len(the_coord[0])))*np.nan
        the_fcp_min[0] = the_coord[0]
        for t in range(len(the_coord[0])):
            if np.any((the_mag[0][::th_avg_fs] >= the_coord[0][t]) & (the_mag[0][::th_avg_fs] < the_coord[0][t]+60)):
                the_fcp_min[1][t] = np.nanmean(the_fcp_arr[np.where((the_mag[0][::th_avg_fs] >= the_coord[0][t]) & (the_mag[0][::th_avg_fs] < the_coord[0][t+1]))[0]])

        the_tind = np.where((the_coord[0] >= trange[0]) & (the_coord[0] <= trange[1]))[0]
        the_peaks_tind = [[],[],[],[]]
        for l in range(the_n_peaks):
            if np.isnan(the_peaks_times[0,l]): continue 
            else:
                start = (np.abs(the_coord[0] - the_peaks_times[0,l])).argmin()
                end = (np.abs(the_coord[0] - the_peaks_times[1,l])).argmin()
                for ind in range(start,end+1): 
                    the_peaks_tind[0].append(ind-the_tind[0])
                    the_peaks_tind[1].append(the_coord[0][ind])
                    the_peaks_tind[2].append(np.nan)
                    the_peaks_tind[3].append(np.nanmean(the_peaks[1][np.where(the_peaks[2] == list(dict.fromkeys(the_peaks[2]))[l])[0]]))
        if len(the_peaks_tind[0]) > 0:
            for m in range(len(the_peaks[0])):
                the_peaks_tind[2][(np.abs(the_peaks[0][m] - the_peaks_tind[1])).argmin()] = the_peaks[1][m]
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_the-peaks', the_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_the-peaks-times', the_peaks_times)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_the-n-peaks', the_n_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_the', the_peaks_tind)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_the-time', the_coord[0][the_tind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_the-coord', the_coord[1][the_tind,:])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_the-l', the_l[1][the_tind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_the-mlt', the_mlt[1][the_tind])
        the_imag_mltind = time_align(the_imag_new[0], the_coord[0], the_imag_new[1])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_the-imag', the_imag_mltind[1][the_tind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_the-fcp', the_fcp_min[1][the_tind])

        # GOES
        goes_f_max = 1.

        g15_fcp_new = time_align(g15_mag[0][::goes_avg_fs], g15_t, g15_fcp_arr)
        g15_t_mask = broadband_mask(g15_spec_new/g15_median, g15_freq, g15_t, g15_fcp_new[1], 'goes')
        g15_peaks_ind, g15_peaks, g15_peaks_times, g15_n_peaks = spec_peaks(g15_t, g15_spec_new/g15_median, g15_freq, goes_fs, g15_fcp_new[1]/16, g15_fcp_new[1], goes_f_max, g15_t_mask)#, g15_l_new[1], goes_l_min, g15_imag_new[1])
        
        g15_l_mltind = time_align(g15_l[0], rbspa_mlt[0], g15_l[1])
        if np.any(np.diff(g15_mlt[1]) < 0): 
            for zero_crossing in np.where(np.diff(g15_mlt[1]) < 0)[0] + 1: #+1 because np.where finds the last place that the MLT was < 24; want the first place that it's > 0
                g15_mlt[1][zero_crossing:] += 24 #to avoid interpolation issues when the MLT crosses from 23.9 to 0
        g15_mlt_new = time_align(g15_mlt[0], rbspa_mlt[0], g15_mlt[1])
        g15_mlt_new[1][np.where(g15_mlt_new[1] >= 5*24)] -= 5*24 #to get back to 0-24 MLT values; not strictly necessary
        g15_mlt_new[1][np.where(g15_mlt_new[1] >= 4*24)] -= 4*24 #longest storm is 4.3 days, so this could have been done up to 5 times
        g15_mlt_new[1][np.where(g15_mlt_new[1] >= 3*24)] -= 3*24
        g15_mlt_new[1][np.where(g15_mlt_new[1] >= 2*24)] -= 2*24
        g15_mlt_new[1][np.where(g15_mlt_new[1] >= 24)] -= 24
        g15_fcp_min = time_align(g15_mag[0][::goes_avg_fs], rbspa_mlt[0], g15_fcp_arr)

        g15_tind = np.where((g15_coord_new[0] >= trange[0]) & (g15_coord_new[0] <= trange[1]))[0]
        g15_mltind = np.where((g15_mlt_new[0] >= trange[0]) & (g15_mlt_new[0] <= trange[1]))[0]
        g15_peaks_tind = []
        g15_peaks_mltind = [[],[],[],[]]
        for l in range(g15_n_peaks): 
            g15_peaks_tind.extend(np.where((g15_coord_new[0] >= g15_peaks_times[0,l]) & (g15_coord_new[0] <= g15_peaks_times[1,l]))[0])
            mlt_start = (np.abs(g15_mlt_new[0] - g15_peaks_times[0,l])).argmin()
            mlt_end = (np.abs(g15_mlt_new[0] - g15_peaks_times[1,l])).argmin()
            for mlt_ind in range(mlt_start,mlt_end+1): 
                g15_peaks_mltind[0].append(mlt_ind-g15_mltind[0])
                g15_peaks_mltind[1].append(g15_mlt_new[0][mlt_ind])
                g15_peaks_mltind[2].append(np.nan)
                g15_peaks_mltind[3].append(np.nanmean(g15_peaks[1][np.where(g15_peaks[2] == list(dict.fromkeys(g15_peaks[2]))[l])[0]]))
        if len(g15_peaks_tind) > 0:
            for m in range(len(g15_peaks[0])):
                g15_peaks_mltind[2][(np.abs(g15_peaks[0][m] - g15_peaks_mltind[1])).argmin()] = g15_peaks[1][m]
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_g15-peaks', g15_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_g15-peaks-times', g15_peaks_times)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_g15-n-peaks', g15_n_peaks)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_g15', g15_peaks_tind-g15_tind[0])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_g15-time-coord', g15_coord_new[0][g15_tind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_g15-coord', g15_coord_new[1][g15_tind,:])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_g15-time', g15_mlt_new[0][g15_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_g15-mltind', g15_peaks_mltind)
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_g15-l', g15_l_mltind[1][g15_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_g15-mlt', g15_mlt_new[1][g15_mltind])
        g15_imag_mltind = time_align(g15_imag_new[0], g15_mlt_new[0], g15_imag_new[1])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_g15-imag', g15_imag_mltind[1][g15_mltind])
        np.save(file_path+'EMIC Index Files/'+storm_day+'_'+storm_phase+'_g15-fcp', g15_fcp_min[1][g15_mltind])

        # # %% Plotting EMIC identification over time
        
        # Van Allen Probes
        pytplot.store_data('rbspa_waves', data={'x': rbspa_peaks[0], 'y': rbspa_peaks[1]})
        pytplot.tplot_copy('rbspa_fcp', 'rbspa_fcp_EMIC')
        pytplot.tplot_copy('rbspa_fcHe', 'rbspa_fcHe_EMIC')
        pytplot.tplot_copy('rbspa_fcO', 'rbspa_fcO_EMIC')
        pytplot.store_data('rbspa_EMIC', data=['rbspa_waves', 'rbspa_fcp_EMIC', 'rbspa_fcHe_EMIC', 'rbspa_fcO_EMIC'])

        pyspedas.options('rbspa_waves', 'color', 'black')
        pyspedas.options('rbspa_waves', 'symbols', True)
        pyspedas.options('rbspa_fcp_EMIC', 'color', 'black')
        pyspedas.options('rbspa_fcp_EMIC', 'thick', 1)
        pyspedas.options('rbspa_fcHe_EMIC', 'color', 'black')
        pyspedas.options('rbspa_fcHe_EMIC', 'thick', 1)
        pyspedas.options('rbspa_fcO_EMIC', 'color', 'black')
        pyspedas.options('rbspa_fcO_EMIC', 'thick', 1)
        pyspedas.options('rbspa_EMIC', 'yrange', [0,6])
        pyspedas.options('rbspa_EMIC', 'ytitle', 'RBSP\nA\n')

        pytplot.store_data('rbspb_waves', data={'x': rbspb_peaks[0], 'y': rbspb_peaks[1]})
        pytplot.tplot_copy('rbspb_fcp', 'rbspb_fcp_EMIC')
        pytplot.tplot_copy('rbspb_fcHe', 'rbspb_fcHe_EMIC')
        pytplot.tplot_copy('rbspb_fcO', 'rbspb_fcO_EMIC')
        pytplot.store_data('rbspb_EMIC', data=['rbspb_waves', 'rbspb_fcp_EMIC', 'rbspb_fcHe_EMIC', 'rbspb_fcO_EMIC'])

        pyspedas.options('rbspb_waves', 'color', 'black')
        pyspedas.options('rbspb_waves', 'symbols', True)
        pyspedas.options('rbspb_fcp_EMIC', 'color', 'black')
        pyspedas.options('rbspb_fcp_EMIC', 'thick', 1)
        pyspedas.options('rbspb_fcHe_EMIC', 'color', 'black')
        pyspedas.options('rbspb_fcHe_EMIC', 'thick', 1)
        pyspedas.options('rbspb_fcO_EMIC', 'color', 'black')
        pyspedas.options('rbspb_fcO_EMIC', 'thick', 1)
        pyspedas.options('rbspb_EMIC', 'yrange', [0,6])
        pyspedas.options('rbspb_EMIC', 'ytitle', 'RBSP\nB\n')
        
        # MMS
        pytplot.store_data('mms1_waves', data={'x': mms1_peaks[0], 'y': mms1_peaks[1]})
        pytplot.tplot_copy('mms1_fcp', 'mms1_fcp_EMIC')
        pytplot.tplot_copy('mms1_fcHe', 'mms1_fcHe_EMIC')
        pytplot.tplot_copy('mms1_fcO', 'mms1_fcO_EMIC')
        pytplot.store_data('mms1_EMIC', data=['mms1_waves', 'mms1_fcp_EMIC', 'mms1_fcHe_EMIC', 'mms1_fcO_EMIC'])

        pyspedas.options('mms1_waves', 'color', 'black')
        pyspedas.options('mms1_waves', 'symbols', True)
        pyspedas.options('mms1_fcp_EMIC', 'color', 'black')
        pyspedas.options('mms1_fcp_EMIC', 'thick', 1)
        pyspedas.options('mms1_fcHe_EMIC', 'color', 'black')
        pyspedas.options('mms1_fcHe_EMIC', 'thick', 1)
        pyspedas.options('mms1_fcO_EMIC', 'color', 'black')
        pyspedas.options('mms1_fcO_EMIC', 'thick', 1)
        pyspedas.options('mms1_EMIC', 'yrange', [0,4])
        pyspedas.options('mms1_EMIC', 'ytitle', 'MMS\n1\n')

        pytplot.store_data('mms2_waves', data={'x': mms2_peaks[0], 'y': mms2_peaks[1]})
        pytplot.tplot_copy('mms2_fcp', 'mms2_fcp_EMIC')
        pytplot.tplot_copy('mms2_fcHe', 'mms2_fcHe_EMIC')
        pytplot.tplot_copy('mms2_fcO', 'mms2_fcO_EMIC')
        pytplot.store_data('mms2_EMIC', data=['mms2_waves', 'mms2_fcp_EMIC', 'mms2_fcHe_EMIC', 'mms2_fcO_EMIC'])

        pyspedas.options('mms2_waves', 'color', 'black')
        pyspedas.options('mms2_waves', 'symbols', True)
        pyspedas.options('mms2_fcp_EMIC', 'color', 'black')
        pyspedas.options('mms2_fcp_EMIC', 'thick', 1)
        pyspedas.options('mms2_fcHe_EMIC', 'color', 'black')
        pyspedas.options('mms2_fcHe_EMIC', 'thick', 1)
        pyspedas.options('mms2_fcO_EMIC', 'color', 'black')
        pyspedas.options('mms2_fcO_EMIC', 'thick', 1)
        pyspedas.options('mms2_EMIC', 'yrange', [0,4])
        pyspedas.options('mms2_EMIC', 'ytitle', 'MMS\n2\n')

        pytplot.store_data('mms3_waves', data={'x': mms3_peaks[0], 'y': mms3_peaks[1]})
        pytplot.tplot_copy('mms3_fcp', 'mms3_fcp_EMIC')
        pytplot.tplot_copy('mms3_fcHe', 'mms3_fcHe_EMIC')
        pytplot.tplot_copy('mms3_fcO', 'mms3_fcO_EMIC')
        pytplot.store_data('mms3_EMIC', data=['mms3_waves', 'mms3_fcp_EMIC', 'mms3_fcHe_EMIC', 'mms3_fcO_EMIC'])

        pyspedas.options('mms3_waves', 'color', 'black')
        pyspedas.options('mms3_waves', 'symbols', True)
        pyspedas.options('mms3_fcp_EMIC', 'color', 'black')
        pyspedas.options('mms3_fcp_EMIC', 'thick', 1)
        pyspedas.options('mms3_fcHe_EMIC', 'color', 'black')
        pyspedas.options('mms3_fcHe_EMIC', 'thick', 1)
        pyspedas.options('mms3_fcO_EMIC', 'color', 'black')
        pyspedas.options('mms3_fcO_EMIC', 'thick', 1)
        pyspedas.options('mms3_EMIC', 'yrange', [0,4])
        pyspedas.options('mms3_EMIC', 'ytitle', 'MMS\n3\n')

        pytplot.store_data('mms4_waves', data={'x': mms4_peaks[0], 'y': mms4_peaks[1]})
        pytplot.tplot_copy('mms4_fcp', 'mms4_fcp_EMIC')
        pytplot.tplot_copy('mms4_fcHe', 'mms4_fcHe_EMIC')
        pytplot.tplot_copy('mms4_fcO', 'mms4_fcO_EMIC')
        pytplot.store_data('mms4_EMIC', data=['mms4_waves', 'mms4_fcp_EMIC', 'mms4_fcHe_EMIC', 'mms4_fcO_EMIC'])

        pyspedas.options('mms4_waves', 'color', 'black')
        pyspedas.options('mms4_waves', 'symbols', True)
        pyspedas.options('mms4_fcp_EMIC', 'color', 'black')
        pyspedas.options('mms4_fcp_EMIC', 'thick', 1)
        pyspedas.options('mms4_fcHe_EMIC', 'color', 'black')
        pyspedas.options('mms4_fcHe_EMIC', 'thick', 1)
        pyspedas.options('mms4_fcO_EMIC', 'color', 'black')
        pyspedas.options('mms4_fcO_EMIC', 'thick', 1)
        pyspedas.options('mms4_EMIC', 'yrange', [0,4])
        pyspedas.options('mms4_EMIC', 'ytitle', 'MMS\n4\n           Frequency [Hz]')

        # THEMIS
        pytplot.store_data('tha_waves', data={'x': tha_peaks[0], 'y': tha_peaks[1]})
        pytplot.tplot_copy('tha_fcp', 'tha_fcp_EMIC')
        pytplot.tplot_copy('tha_fcHe', 'tha_fcHe_EMIC')
        pytplot.tplot_copy('tha_fcO', 'tha_fcO_EMIC')
        pytplot.store_data('tha_EMIC', data=['tha_waves', 'tha_fcp_EMIC', 'tha_fcHe_EMIC', 'tha_fcO_EMIC'])

        pyspedas.options('tha_waves', 'color', 'black')
        pyspedas.options('tha_waves', 'symbols', True)
        pyspedas.options('tha_fcp_EMIC', 'color', 'black')
        pyspedas.options('tha_fcp_EMIC', 'thick', 1)
        pyspedas.options('tha_fcHe_EMIC', 'color', 'black')
        pyspedas.options('tha_fcHe_EMIC', 'thick', 1)
        pyspedas.options('tha_fcO_EMIC', 'color', 'black')
        pyspedas.options('tha_fcO_EMIC', 'thick', 1)
        pyspedas.options('tha_EMIC', 'yrange', [0,2])
        pyspedas.options('tha_EMIC', 'charsize', 10)
        pyspedas.options('tha_EMIC', 'ytitle', 'THEMIS\nA\n\n')

        pytplot.store_data('thd_waves', data={'x': thd_peaks[0], 'y': thd_peaks[1]})
        pytplot.tplot_copy('thd_fcp', 'thd_fcp_EMIC')
        pytplot.tplot_copy('thd_fcHe', 'thd_fcHe_EMIC')
        pytplot.tplot_copy('thd_fcO', 'thd_fcO_EMIC')
        pytplot.store_data('thd_EMIC', data=['thd_waves', 'thd_fcp_EMIC', 'thd_fcHe_EMIC', 'thd_fcO_EMIC'])

        pyspedas.options('thd_waves', 'color', 'black')
        pyspedas.options('thd_waves', 'symbols', True)
        pyspedas.options('thd_fcp_EMIC', 'color', 'black')
        pyspedas.options('thd_fcp_EMIC', 'thick', 1)
        pyspedas.options('thd_fcHe_EMIC', 'color', 'black')
        pyspedas.options('thd_fcHe_EMIC', 'thick', 1)
        pyspedas.options('thd_fcO_EMIC', 'color', 'black')
        pyspedas.options('thd_fcO_EMIC', 'thick', 1)
        pyspedas.options('thd_EMIC', 'yrange', [0,2])
        pyspedas.options('thd_EMIC', 'charsize', 10)
        pyspedas.options('thd_EMIC', 'ytitle', 'THEMIS\nD\n\n')

        pytplot.store_data('the_waves', data={'x': the_peaks[0], 'y': the_peaks[1]})
        pytplot.tplot_copy('the_fcp', 'the_fcp_EMIC')
        pytplot.tplot_copy('the_fcHe', 'the_fcHe_EMIC')
        pytplot.tplot_copy('the_fcO', 'the_fcO_EMIC')
        pytplot.store_data('the_EMIC', data=['the_waves', 'the_fcp_EMIC', 'the_fcHe_EMIC', 'the_fcO_EMIC'])

        pyspedas.options('the_waves', 'color', 'black')
        pyspedas.options('the_waves', 'symbols', True)
        pyspedas.options('the_fcp_EMIC', 'color', 'black')
        pyspedas.options('the_fcp_EMIC', 'thick', 1)
        pyspedas.options('the_fcHe_EMIC', 'color', 'black')
        pyspedas.options('the_fcHe_EMIC', 'thick', 1)
        pyspedas.options('the_fcO_EMIC', 'color', 'black')
        pyspedas.options('the_fcO_EMIC', 'thick', 1)
        pyspedas.options('the_EMIC', 'yrange', [0,2])
        pyspedas.options('the_EMIC', 'charsize', 10)
        pyspedas.options('the_EMIC', 'ytitle', 'THEMIS\nE\n\n')

        # GOES
        pytplot.store_data('g15_waves', data={'x': g15_peaks[0], 'y': g15_peaks[1]})
        pytplot.tplot_copy('g15_fcp', 'g15_fcp_EMIC')
        pytplot.tplot_copy('g15_fcHe', 'g15_fcHe_EMIC')
        pytplot.tplot_copy('g15_fcO', 'g15_fcO_EMIC')
        pytplot.store_data('g15_EMIC', data=['g15_waves', 'g15_fcp_EMIC', 'g15_fcHe_EMIC', 'g15_fcO_EMIC'])

        pyspedas.options('g15_waves', 'color', 'black')
        pyspedas.options('g15_waves', 'symbols', True)
        pyspedas.options('g15_fcp_EMIC', 'color', 'black')
        pyspedas.options('g15_fcp_EMIC', 'thick', 1)
        pyspedas.options('g15_fcHe_EMIC', 'color', 'black')
        pyspedas.options('g15_fcHe_EMIC', 'thick', 1)
        pyspedas.options('g15_fcO_EMIC', 'color', 'black')
        pyspedas.options('g15_fcO_EMIC', 'thick', 1)
        pyspedas.options('g15_EMIC', 'yrange', [0,1])
        pyspedas.options('g15_EMIC', 'ytitle', '\nGOES\n15\n')


        pyspedas.tplot_options('xmargin', [0.15,0.15]) #left/right margins, inches
        pyspedas.tplot_options('axis_font_size', 9)
        pytplot.xlim(trange[0], trange[1])
        pytplot.tplot_options('title', storm_day+' '+storm_phase+' Phase EMIC Waves')
        pytplot.tplot(['rbspa_EMIC', 'rbspb_EMIC', \
                        'mms1_EMIC', 'mms2_EMIC', 'mms3_EMIC', 'mms4_EMIC', \
                            'tha_EMIC', 'thd_EMIC', 'the_EMIC', 'g15_EMIC'], \
                              vert_spacing=30, dpi=150, \
                                  save_png=file_path+storm_day+'_'+storm_phase+'_EMIC')
        
        # # %% Plotting EMIC identificaiton on orbits
        
        plt.figure(figsize=(10,10), dpi=150)
        max_gsm = 12 #Re, maximum distance along each axis

        # Van Allen Probes
        if rbspa_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'

        plt.plot(rbspa_coord[1][rbspa_tind,0]/Re, rbspa_coord[1][rbspa_tind,1]/Re, label='Van Allen Probes', color=color)
        plt.plot(rbspa_coord[1][rbspa_tind,0][0]/Re, rbspa_coord[1][rbspa_tind,1][0]/Re, 'o', color=color)
        plt.plot(rbspa_coord[1][rbspa_tind,0][-1]/Re, rbspa_coord[1][rbspa_tind,1][-1]/Re, '*', color=color, markersize=15)
        if (rbspa_coord[1][rbspa_tind,0][0]/Re - 0.2 > -max_gsm) and (rbspa_coord[1][rbspa_tind,0][0]/Re - 0.2 < max_gsm) and (rbspa_coord[1][rbspa_tind,1][0]/Re + 1.25 > -max_gsm) and (rbspa_coord[1][rbspa_tind,1][0]/Re + 1.25 < max_gsm):
            plt.text(rbspa_coord[1][rbspa_tind,0][0]/Re - 0.2, rbspa_coord[1][rbspa_tind,1][0]/Re + 1.25, 'VA-A', fontsize='x-large')
        elif np.any(rbspa_coord[1][rbspa_tind,0]/Re > -max_gsm) and np.any(rbspa_coord[1][rbspa_tind,0]/Re < max_gsm) and np.any(rbspa_coord[1][rbspa_tind,1]/Re > -max_gsm) and np.any(rbspa_coord[1][rbspa_tind,1]/Re < max_gsm):
            coord = np.where((np.abs(rbspa_coord[1][rbspa_tind,0])/Re < max_gsm) & (np.abs(rbspa_coord[1][rbspa_tind,1])/Re < max_gsm))[0][0]
            plt.text(rbspa_coord[1][rbspa_tind,0][coord]/Re, rbspa_coord[1][rbspa_tind,1][coord]/Re, 'VA-A', fontsize='x-large')

        if len(rbspa_peaks_tind[0]) > 0:
            plt.plot(rbspa_coord[1][rbspa_peaks_tind[0]+rbspa_tind[0],0]/Re, rbspa_coord[1][rbspa_peaks_tind[0]+rbspa_tind[0],1]/Re, '.', color='orange', markersize=10)

        if rbspb_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'

        plt.plot(rbspb_coord[1][rbspb_tind,0]/Re, rbspb_coord[1][rbspb_tind,1]/Re, color=color)
        plt.plot(rbspb_coord[1][rbspb_tind,0][0]/Re, rbspb_coord[1][rbspb_tind,1][0]/Re, 'o', color=color)
        plt.plot(rbspb_coord[1][rbspb_tind,0][-1]/Re, rbspb_coord[1][rbspb_tind,1][-1]/Re, '*', color=color, markersize=15)
        if (rbspb_coord[1][rbspb_tind,0][0]/Re - 0.2 > -max_gsm) and (rbspb_coord[1][rbspb_tind,0][0]/Re - 0.2 < max_gsm) and (rbspb_coord[1][rbspb_tind,1][0]/Re + 1.25 > -max_gsm) and (rbspb_coord[1][rbspb_tind,1][0]/Re + 1.25 < max_gsm):
            plt.text(rbspb_coord[1][rbspb_tind,0][0]/Re - 0.2, rbspb_coord[1][rbspb_tind,1][0]/Re + 1.25, 'VA-B', fontsize='x-large')
        elif np.any(rbspb_coord[1][rbspb_tind,0]/Re > -max_gsm) and np.any(rbspb_coord[1][rbspb_tind,0]/Re < max_gsm) and np.any(rbspb_coord[1][rbspb_tind,1]/Re > -max_gsm) and np.any(rbspb_coord[1][rbspb_tind,1]/Re < max_gsm):
            coord = np.where((np.abs(rbspb_coord[1][rbspb_tind,0])/Re < max_gsm) & (np.abs(rbspb_coord[1][rbspb_tind,1])/Re < max_gsm))[0][0]
            plt.text(rbspb_coord[1][rbspb_tind,0][coord]/Re, rbspb_coord[1][rbspb_tind,1][coord]/Re, 'VA-B', fontsize='x-large')

        if len(rbspb_peaks_tind[0]) > 0:
            plt.plot(rbspb_coord[1][rbspb_peaks_tind[0]+rbspb_tind[0],0]/Re, rbspb_coord[1][rbspb_peaks_tind[0]+rbspb_tind[0],1]/Re, '.', color='orange', markersize=10)

        # MMS
        if mms1_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'

        plt.plot(mms1_coord[1][mms1_tind,0]/Re, mms1_coord[1][mms1_tind,1]/Re, label='MMS Probes', color=color)
        plt.plot(mms1_coord[1][mms1_tind,0][0]/Re, mms1_coord[1][mms1_tind,1][0]/Re, 'o', color=color)
        plt.plot(mms1_coord[1][mms1_tind,0][-1]/Re, mms1_coord[1][mms1_tind,1][-1]/Re, '*', color=color, markersize=15)
        if (mms1_coord[1][mms1_tind,0][0]/Re - 0.2 > -max_gsm) and (mms1_coord[1][mms1_tind,0][0]/Re - 0.2 < max_gsm) and (mms1_coord[1][mms1_tind,1][0]/Re + 1.25 > -max_gsm) and (mms1_coord[1][mms1_tind,1][0]/Re + 1.25 < max_gsm):
            plt.text(mms1_coord[1][mms1_tind,0][0]/Re - 0.2, mms1_coord[1][mms1_tind,1][0]/Re + 1.25, 'MMS-1', fontsize='x-large')
        elif np.any(mms1_coord[1][mms1_tind,0]/Re > -max_gsm) and np.any(mms1_coord[1][mms1_tind,0]/Re < max_gsm) and np.any(mms1_coord[1][mms1_tind,1]/Re > -max_gsm) and np.any(mms1_coord[1][mms1_tind,1]/Re < max_gsm):
            coord = np.where((np.abs(mms1_coord[1][mms1_tind,0])/Re < max_gsm) & (np.abs(mms1_coord[1][mms1_tind,1])/Re < max_gsm))[0][0]
            plt.text(mms1_coord[1][mms1_tind,0][coord]/Re, mms1_coord[1][mms1_tind,1][coord]/Re, 'MMS-1', fontsize='x-large')

        if len(mms1_peaks_tind[0]) > 0:
            plt.plot(mms1_coord[1][mms1_peaks_tind[0]+mms1_tind[0],0]/Re, mms1_coord[1][mms1_peaks_tind[0]+mms1_tind[0],1]/Re, '.', color='orange', markersize=10)

        if mms2_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'

        plt.plot(mms2_coord[1][mms2_tind,0]/Re, mms2_coord[1][mms2_tind,1]/Re, color=color)
        plt.plot(mms2_coord[1][mms2_tind,0][0]/Re, mms2_coord[1][mms2_tind,1][0]/Re, 'o', color=color)
        plt.plot(mms2_coord[1][mms2_tind,0][-1]/Re, mms2_coord[1][mms2_tind,1][-1]/Re, '*', color=color, markersize=15)
        if (mms2_coord[1][mms2_tind,0][0]/Re - 0.2 > -max_gsm) and (mms2_coord[1][mms2_tind,0][0]/Re - 0.2 < max_gsm) and (mms2_coord[1][mms2_tind,1][0]/Re + 1.25 > -max_gsm) and (mms2_coord[1][mms2_tind,1][0]/Re + 1.25 < max_gsm):
            plt.text(mms2_coord[1][mms2_tind,0][0]/Re - 0.2, mms2_coord[1][mms2_tind,1][0]/Re + 1.25, 'MMS-2', fontsize='x-large')
        elif np.any(mms2_coord[1][mms2_tind,0]/Re > -max_gsm) and np.any(mms2_coord[1][mms2_tind,0]/Re < max_gsm) and np.any(mms2_coord[1][mms2_tind,1]/Re > -max_gsm) and np.any(mms2_coord[1][mms2_tind,1]/Re < max_gsm):
            coord = np.where((np.abs(mms2_coord[1][mms2_tind,0])/Re < max_gsm) & (np.abs(mms2_coord[1][mms2_tind,1])/Re < max_gsm))[0][0]
            plt.text(mms2_coord[1][mms2_tind,0][coord]/Re, mms2_coord[1][mms2_tind,1][coord]/Re, 'MMS-2', fontsize='x-large')

        if len(mms2_peaks_tind[0]) > 0:
            plt.plot(mms2_coord[1][mms2_peaks_tind[0]+mms2_tind[0],0]/Re, mms2_coord[1][mms2_peaks_tind[0]+mms2_tind[0],1]/Re, '.', color='orange', markersize=10)

        if mms3_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'

        plt.plot(mms3_coord[1][mms3_tind,0]/Re, mms3_coord[1][mms3_tind,1]/Re, color=color)
        plt.plot(mms3_coord[1][mms3_tind,0][0]/Re, mms3_coord[1][mms3_tind,1][0]/Re, 'o', color=color)
        plt.plot(mms3_coord[1][mms3_tind,0][-1]/Re, mms3_coord[1][mms3_tind,1][-1]/Re, '*', color=color, markersize=15)
        if (mms3_coord[1][mms3_tind,0][0]/Re - 0.2 > -max_gsm) and (mms3_coord[1][mms3_tind,0][0]/Re - 0.2 < max_gsm) and (mms3_coord[1][mms3_tind,1][0]/Re + 1.25 > -max_gsm) and (mms3_coord[1][mms3_tind,1][0]/Re + 1.25 < max_gsm):
            plt.text(mms3_coord[1][mms3_tind,0][0]/Re - 0.2, mms3_coord[1][mms3_tind,1][0]/Re + 1.25, 'MMS-3', fontsize='x-large')
        elif np.any(mms3_coord[1][mms3_tind,0]/Re > -max_gsm) and np.any(mms3_coord[1][mms3_tind,0]/Re < max_gsm) and np.any(mms3_coord[1][mms3_tind,1]/Re > -max_gsm) and np.any(mms3_coord[1][mms3_tind,1]/Re < max_gsm):
            coord = np.where((np.abs(mms3_coord[1][mms3_tind,0])/Re < max_gsm) & (np.abs(mms3_coord[1][mms3_tind,1])/Re < max_gsm))[0][0]
            plt.text(mms3_coord[1][mms3_tind,0][coord]/Re, mms3_coord[1][mms3_tind,1][coord]/Re, 'MMS-3', fontsize='x-large')

        if len(mms3_peaks_tind[0]) > 0:
            plt.plot(mms3_coord[1][mms3_peaks_tind[0]+mms3_tind[0],0]/Re, mms3_coord[1][mms3_peaks_tind[0]+mms3_tind[0],1]/Re, '.', color='orange', markersize=10)

        if mms4_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'

        plt.plot(mms4_coord[1][mms4_tind,0]/Re, mms4_coord[1][mms4_tind,1]/Re, color=color)
        plt.plot(mms4_coord[1][mms4_tind,0][0]/Re, mms4_coord[1][mms4_tind,1][0]/Re, 'o', color=color)
        plt.plot(mms4_coord[1][mms4_tind,0][-1]/Re, mms4_coord[1][mms4_tind,1][-1]/Re, '*', color=color, markersize=15)
        if (mms4_coord[1][mms4_tind,0][0]/Re - 0.2 > -max_gsm) and (mms4_coord[1][mms4_tind,0][0]/Re - 0.2 < max_gsm) and (mms4_coord[1][mms4_tind,1][0]/Re + 1.25 > -max_gsm) and (mms4_coord[1][mms4_tind,1][0]/Re + 1.25 < max_gsm):
            plt.text(mms4_coord[1][mms4_tind,0][0]/Re - 0.2, mms4_coord[1][mms4_tind,1][0]/Re + 1.25, 'MMS-4', fontsize='x-large')
        elif np.any(mms4_coord[1][mms4_tind,0]/Re > -max_gsm) and np.any(mms4_coord[1][mms4_tind,0]/Re < max_gsm) and np.any(mms4_coord[1][mms4_tind,1]/Re > -max_gsm) and np.any(mms4_coord[1][mms4_tind,1]/Re < max_gsm):
            coord = np.where((np.abs(mms4_coord[1][mms4_tind,0])/Re < max_gsm) & (np.abs(mms4_coord[1][mms4_tind,1])/Re < max_gsm))[0][0]
            plt.text(mms4_coord[1][mms4_tind,0][coord]/Re, mms4_coord[1][mms4_tind,1][coord]/Re, 'MMS-4', fontsize='x-large')

        if len(mms4_peaks_tind[0]) > 0:
            plt.plot(mms4_coord[1][mms4_peaks_tind[0]+mms4_tind[0],0]/Re, mms4_coord[1][mms4_peaks_tind[0]+mms4_tind[0],1]/Re, '.', color='orange', markersize=10)

        # THEMIS
        if tha_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'

        plt.plot(tha_coord[1][tha_tind,0], tha_coord[1][tha_tind,1], label='THEMIS Probes', color=color)
        plt.plot(tha_coord[1][tha_tind,0][0], tha_coord[1][tha_tind,1][0], 'o', color=color)
        plt.plot(tha_coord[1][tha_tind,0][-1], tha_coord[1][tha_tind,1][-1], '*', color=color, markersize=15)
        if (tha_coord[1][tha_tind,0][0] - 0.2 > -max_gsm) and (tha_coord[1][tha_tind,0][0] - 0.2 < max_gsm) and (tha_coord[1][tha_tind,1][0] + 1.25 > -max_gsm) and (tha_coord[1][tha_tind,1][0] + 1.25 < max_gsm):
            plt.text(tha_coord[1][tha_tind,0][0] - 0.2, tha_coord[1][tha_tind,1][0] + 1.25, 'TH-A', fontsize='x-large')
        elif np.any(tha_coord[1][tha_tind,0] > -max_gsm) and np.any(tha_coord[1][tha_tind,0] < max_gsm) and np.any(tha_coord[1][tha_tind,1] > -max_gsm) and np.any(tha_coord[1][tha_tind,1] < max_gsm):
            coord = np.where((np.abs(tha_coord[1][tha_tind,0]) < max_gsm) & (np.abs(tha_coord[1][tha_tind,1]) < max_gsm))[0][0]
            plt.text(tha_coord[1][tha_tind,0][coord], tha_coord[1][tha_tind,1][coord], 'TH-A', fontsize='x-large')

        if len(tha_peaks_tind[0]) > 0:
            plt.plot(tha_coord[1][tha_peaks_tind[0]+tha_tind[0],0], tha_coord[1][tha_peaks_tind[0]+tha_tind[0],1], '.', color='orange', markersize=10)

        if thd_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'

        plt.plot(thd_coord[1][thd_tind,0], thd_coord[1][thd_tind,1], color=color)
        plt.plot(thd_coord[1][thd_tind,0][0], thd_coord[1][thd_tind,1][0], 'o', color=color)
        plt.plot(thd_coord[1][thd_tind,0][-1], thd_coord[1][thd_tind,1][-1], '*', color=color, markersize=15)
        if (thd_coord[1][thd_tind,0][0] - 0.2 > -max_gsm) and (thd_coord[1][thd_tind,0][0] - 0.2 < max_gsm) and (thd_coord[1][thd_tind,1][0] + 1.25 > -max_gsm) and (thd_coord[1][thd_tind,1][0] + 1.25 < max_gsm):
            plt.text(thd_coord[1][thd_tind,0][0] - 0.2, thd_coord[1][thd_tind,1][0] + 1.25, 'TH-D', fontsize='x-large')
        elif np.any(thd_coord[1][thd_tind,0] > -max_gsm) and np.any(thd_coord[1][thd_tind,0] < max_gsm) and np.any(thd_coord[1][thd_tind,1] > -max_gsm) and np.any(thd_coord[1][thd_tind,1] < max_gsm):
            coord = np.where((np.abs(thd_coord[1][thd_tind,0]) < max_gsm) & (np.abs(thd_coord[1][thd_tind,1]) < max_gsm))[0][0]
            plt.text(thd_coord[1][thd_tind,0][coord], thd_coord[1][thd_tind,1][coord], 'TH-D', fontsize='x-large')

        if len(thd_peaks_tind[0]) > 0:
            plt.plot(thd_coord[1][thd_peaks_tind[0]+thd_tind[0],0], thd_coord[1][thd_peaks_tind[0]+thd_tind[0],1], '.', color='orange', markersize=10)

        if the_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'

        plt.plot(the_coord[1][the_tind,0], the_coord[1][the_tind,1], color=color)
        plt.plot(the_coord[1][the_tind,0][0], the_coord[1][the_tind,1][0], 'o', color=color)
        plt.plot(the_coord[1][the_tind,0][-1], the_coord[1][the_tind,1][-1], '*', color=color, markersize=15)
        if (the_coord[1][the_tind,0][0] - 0.2 > -max_gsm) and (the_coord[1][the_tind,0][0] - 0.2 < max_gsm) and (the_coord[1][the_tind,1][0] + 1.25 > -max_gsm) and (the_coord[1][the_tind,1][0] + 1.25 < max_gsm):
            plt.text(the_coord[1][the_tind,0][0] - 0.2, the_coord[1][the_tind,1][0] + 1.25, 'TH-E', fontsize='x-large')
        elif np.any(the_coord[1][the_tind,0] > -max_gsm) and np.any(the_coord[1][the_tind,0] < max_gsm) and np.any(the_coord[1][the_tind,1] > -max_gsm) and np.any(the_coord[1][the_tind,1] < max_gsm):
            coord = np.where((np.abs(the_coord[1][the_tind,0]) < max_gsm) & (np.abs(the_coord[1][the_tind,1]) < max_gsm))[0][0]
            plt.text(the_coord[1][the_tind,0][coord], the_coord[1][the_tind,1][coord], 'TH-E', fontsize='x-large')

        if len(the_peaks_tind[0]) > 0:
            plt.plot(the_coord[1][the_peaks_tind[0]+the_tind[0],0], the_coord[1][the_peaks_tind[0]+the_tind[0],1], '.', color='orange', markersize=10)

        # GOES
        if len(g15_peaks[1]) == 0 or g15_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'
        
        plt.plot(g15_coord_new[1][g15_tind,0][np.isfinite(g15_coord_new[1][g15_tind,0])], g15_coord_new[1][g15_tind,1][np.isfinite(g15_coord_new[1][g15_tind,1])], label='GOES', color=color)
        plt.plot(g15_coord_new[1][g15_tind,0][np.isfinite(g15_coord_new[1][g15_tind,0])][0], g15_coord_new[1][g15_tind,1][np.isfinite(g15_coord_new[1][g15_tind,1])][0], 'o', color=color)
        plt.plot(g15_coord_new[1][g15_tind,0][np.isfinite(g15_coord_new[1][g15_tind,0])][-1], g15_coord_new[1][g15_tind,1][np.isfinite(g15_coord_new[1][g15_tind,1])][-1], '*', color=color, markersize=15)
        if (g15_coord_new[1][g15_tind,0][0] - 0.2 > -max_gsm) and (g15_coord_new[1][g15_tind,0][0] - 0.2 < max_gsm) and (g15_coord_new[1][g15_tind,1][0] + 1.25 > -max_gsm) and (g15_coord_new[1][g15_tind,1][0] + 1.25 < max_gsm):
            plt.text(g15_coord_new[1][g15_tind,0][np.isfinite(g15_coord_new[1][g15_tind,0])][0] - 0.2, g15_coord_new[1][g15_tind,1][np.isfinite(g15_coord_new[1][g15_tind,1])][0] + 1.25, 'GOES-15', fontsize='x-large')
        elif np.any(g15_coord_new[1][g15_tind,0][np.isfinite(g15_coord_new[1][g15_tind,0])] > -max_gsm) and np.any(g15_coord_new[1][g15_tind,0][np.isfinite(g15_coord_new[1][g15_tind,0])] < max_gsm) and np.any(g15_coord_new[1][g15_tind,1][np.isfinite(g15_coord_new[1][g15_tind,1])] > -max_gsm) and np.any(g15_coord_new[1][g15_tind,1][np.isfinite(g15_coord_new[1][g15_tind,1])] < max_gsm):
            coord = np.where((np.abs(g15_coord_new[1][g15_tind,0][np.isfinite(g15_coord_new[1][g15_tind,0])]) < max_gsm) & (np.abs(g15_coord_new[1][g15_tind,1][np.isfinite(g15_coord_new[1][g15_tind,1])]) < max_gsm))[0][0]
            plt.text(g15_coord_new[1][g15_tind,0][np.isfinite(g15_coord_new[1][g15_tind,0])][coord], g15_coord_new[1][g15_tind,1][np.isfinite(g15_coord_new[1][g15_tind,1])][coord], 'GOES-15', fontsize='x-large')

        if len(g15_peaks_tind) > 0:
            plt.plot(g15_coord_new[1][g15_peaks_tind,0], g15_coord_new[1][g15_peaks_tind,1], '.', color='orange', markersize=10)

        # Coordinate lines and Earth
        plt.axhline(y=0, color='k', linestyle='dashed')
        plt.axvline(x=0, color='k', linestyle='dashed')
        x = np.linspace(-1,1,100)
        y = np.linspace(-1,1,100)
        X, Y = np.meshgrid(x, y)
        f = X**2 + Y**2 - 1
        mask_black = np.where(X <= 0, f, np.nan)
        mask_white = np.where(X > 0, f, np.nan)
        plt.contourf(X, Y, mask_black, levels=[-np.inf, 0], colors='k')
        plt.contourf(X, Y, mask_white, levels=[-np.inf, 0], colors='w', zorder=2)
        plt.contour(X, Y, f, levels=[0], colors='k')

        # Magnetopause
        plt.plot(mag_avg_x, mag_avg_y, color='blue')
        plt.plot(mag_max_x, mag_max_y, color='lightblue')
        plt.plot(mag_min_x, mag_min_y, color='lightblue')

        # Plot range, titles
        plt.xlim([max_gsm,-max_gsm]) # Re
        plt.ylim([max_gsm,-max_gsm])
        plt.title('Satellite Locations (in GSM); ' + storm_day+' - '+storm_phase)
        plt.xlabel('X [Re]')
        plt.ylabel('Y [Re]')
        plt.savefig(file_path+storm_day+'_'+storm_phase+'_orbits with EMIC.png', bbox_inches='tight', pad_inches=0.25)
        plt.show()
        plt.close()
        
        # # %% Plotting EMIC identification on orbits (L/MLT)

        fig = plt.figure(figsize=(10,10), dpi=150)
        ax = plt.subplot(111, projection='polar')
        ax.set_theta_zero_location('E')

        mlt_labels = np.linspace(0,24,8,dtype=int,endpoint=False)
        mlt_ticks = np.linspace(0,2*np.pi,8,endpoint=False)

        ax.set_xticks(mlt_ticks)
        ax.set_xticklabels(mlt_labels)
        ax.set_rlim(0,12)
        ax.set_yticks(np.linspace(2,12,11,dtype=int))

        mlt_to_rad = (2*np.pi)/24

        # Van Allen Probes
        if rbspa_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'

        ax.plot(rbspa_mlt[1][rbspa_mltind]*mlt_to_rad, rbspa_l[1][rbspa_mltind], color=color)
        ax.plot(rbspa_mlt[1][rbspa_mltind][0]*mlt_to_rad, rbspa_l[1][rbspa_mltind][0], 'o', color=color)
        ax.plot(rbspa_mlt[1][rbspa_mltind][-1]*mlt_to_rad, rbspa_l[1][rbspa_mltind][-1], '*', markersize=15, color=color)
        if rbspa_l[1][rbspa_mltind][0] < 12: 
            ax.text(rbspa_mlt[1][rbspa_mltind][0]*mlt_to_rad+0.1, rbspa_l[1][rbspa_mltind][0]-0.5, 'VA-A', fontsize='x-large', color='k')
        elif np.any(rbspa_l[1][rbspa_tind] < 12):
            ax.text(rbspa_mlt[1][rbspa_mltind][np.where(rbspa_l[1][rbspa_mltind] < 12)[0][0]]*mlt_to_rad, rbspa_l[1][rbspa_mltind][np.where(rbspa_l[1][rbspa_mltind] < 12)[0][0]], 'VA-A', fontsize='x-large', color='k')

        if len(rbspa_peaks_mltind[0]) > 0:
            ax.plot(rbspa_mlt[1][rbspa_peaks_mltind[0]+rbspa_mltind[0]]*mlt_to_rad, rbspa_l[1][rbspa_peaks_mltind[0]+rbspa_mltind[0]], '.', color='orange', markersize=10)

        if rbspb_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'

        ax.plot(rbspb_mlt[1][rbspb_mltind]*mlt_to_rad, rbspb_l[1][rbspb_mltind], color=color)
        ax.plot(rbspb_mlt[1][rbspb_mltind][0]*mlt_to_rad, rbspb_l[1][rbspb_mltind][0], 'o', color=color)
        ax.plot(rbspb_mlt[1][rbspb_mltind][-1]*mlt_to_rad, rbspb_l[1][rbspb_mltind][-1], '*', markersize=15, color=color)
        if rbspb_l[1][rbspb_mltind][0] < 12: 
            ax.text(rbspb_mlt[1][rbspb_mltind][0]*mlt_to_rad+0.1, rbspb_l[1][rbspb_mltind][0]-0.5, 'VA-B', fontsize='x-large', color='k')
        elif np.any(rbspb_l[1][rbspb_tind] < 12):
            ax.text(rbspb_mlt[1][rbspb_mltind][np.where(rbspb_l[1][rbspb_mltind] < 12)[0][0]]*mlt_to_rad, rbspb_l[1][rbspb_mltind][np.where(rbspb_l[1][rbspb_mltind] < 12)[0][0]], 'VA-B', fontsize='x-large', color='k')

        if len(rbspb_peaks_mltind[0]) > 0:
            ax.plot(rbspb_mlt[1][rbspb_peaks_mltind[0]+rbspb_mltind[0]]*mlt_to_rad, rbspb_l[1][rbspb_peaks_mltind[0]+rbspb_mltind[0]], '.', color='orange', markersize=10)

        # MMS
        if mms1_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'

        ax.plot(mms1_mlt[1][mms1_tind]*mlt_to_rad, mms1_l[1][mms1_tind], color=color)
        ax.plot(mms1_mlt[1][mms1_tind][0]*mlt_to_rad, mms1_l[1][mms1_tind][0], 'o', color=color)
        ax.plot(mms1_mlt[1][mms1_tind][-1]*mlt_to_rad, mms1_l[1][mms1_tind][-1], '*', markersize=15, color=color)
        if mms1_l[1][mms1_tind][0] < 12: 
            ax.text(mms1_mlt[1][mms1_tind][0]*mlt_to_rad+0.1, mms1_l[1][mms1_tind][0]-0.5, 'MMS-1', fontsize='x-large', color='k')
        elif np.any(mms1_l[1][mms1_tind] < 12):
            ax.text(mms1_mlt[1][mms1_tind][np.where(mms1_l[1][mms1_tind] < 12)[0][0]]*mlt_to_rad, mms1_l[1][mms1_tind][np.where(mms1_l[1][mms1_tind] < 12)[0][0]], 'MMS-1', fontsize='x-large', color='k')

        if len(mms1_peaks_mltind[0]) > 0:
            ax.plot(mms1_mlt[1][mms1_peaks_tind[0]+mms1_tind[0]]*mlt_to_rad, mms1_l[1][mms1_peaks_tind[0]+mms1_tind[0]], '.', color='orange', markersize=10)

        if mms2_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'

        ax.plot(mms2_mlt[1][mms2_tind]*mlt_to_rad, mms2_l[1][mms2_tind], color=color)
        ax.plot(mms2_mlt[1][mms2_tind][0]*mlt_to_rad, mms2_l[1][mms2_tind][0], 'o', color=color)
        ax.plot(mms2_mlt[1][mms2_tind][-1]*mlt_to_rad, mms2_l[1][mms2_tind][-1], '*', markersize=15, color=color)
        if mms2_l[1][mms2_tind][0] < 12: 
            ax.text(mms2_mlt[1][mms2_tind][0]*mlt_to_rad+0.1, mms2_l[1][mms2_tind][0]-0.5, 'MMS-2', fontsize='x-large', color='k')
        elif np.any(mms2_l[1][mms2_tind] < 12):
            ax.text(mms2_mlt[1][mms2_tind][np.where(mms2_l[1][mms2_tind] < 12)[0][0]]*mlt_to_rad, mms2_l[1][mms2_tind][np.where(mms2_l[1][mms2_tind] < 12)[0][0]], 'MMS-2', fontsize='x-large', color='k')

        if len(mms2_peaks_mltind[0]) > 0:
            ax.plot(mms2_mlt[1][mms2_peaks_tind[0]+mms2_tind[0]]*mlt_to_rad, mms2_l[1][mms2_peaks_tind[0]+mms2_tind[0]], '.', color='orange', markersize=10)

        if mms3_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'

        ax.plot(mms3_mlt[1][mms3_tind]*mlt_to_rad, mms3_l[1][mms3_tind], color=color)
        ax.plot(mms3_mlt[1][mms3_tind][0]*mlt_to_rad, mms3_l[1][mms3_tind][0], 'o', color=color)
        ax.plot(mms3_mlt[1][mms3_tind][-1]*mlt_to_rad, mms3_l[1][mms3_tind][-1], '*', markersize=15, color=color)
        if mms3_l[1][mms3_tind][0] < 12: 
            ax.text(mms3_mlt[1][mms3_tind][0]*mlt_to_rad+0.1, mms3_l[1][mms3_tind][0]-0.5, 'MMS-3', fontsize='x-large', color='k')
        elif np.any(mms3_l[1][mms3_tind] < 12):
            ax.text(mms3_mlt[1][mms3_tind][np.where(mms3_l[1][mms3_tind] < 12)[0][0]]*mlt_to_rad, mms3_l[1][mms3_tind][np.where(mms3_l[1][mms3_tind] < 12)[0][0]], 'MMS-3', fontsize='x-large', color='k')

        if len(mms3_peaks_mltind[0]) > 0:
            ax.plot(mms3_mlt[1][mms3_peaks_tind[0]+mms3_tind[0]]*mlt_to_rad, mms3_l[1][mms3_peaks_tind[0]+mms3_tind[0]], '.', color='orange', markersize=10)

        if mms4_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'

        ax.plot(mms4_mlt[1][mms4_tind]*mlt_to_rad, mms4_l[1][mms4_tind], color=color)
        ax.plot(mms4_mlt[1][mms4_tind][0]*mlt_to_rad, mms4_l[1][mms4_tind][0], 'o', color=color)
        ax.plot(mms4_mlt[1][mms4_tind][-1]*mlt_to_rad, mms4_l[1][mms4_tind][-1], '*', markersize=15, color=color)
        if mms4_l[1][mms4_tind][0] < 12: 
            ax.text(mms4_mlt[1][mms4_tind][0]*mlt_to_rad+0.1, mms4_l[1][mms4_tind][0]-0.5, 'MMS-4', fontsize='x-large', color='k')
        elif np.any(mms4_l[1][mms4_tind] < 12):
            ax.text(mms4_mlt[1][mms4_tind][np.where(mms4_l[1][mms4_tind] < 12)[0][0]]*mlt_to_rad, mms4_l[1][mms4_tind][np.where(mms4_l[1][mms4_tind] < 12)[0][0]], 'MMS-4', fontsize='x-large', color='k')

        if len(mms4_peaks_mltind[0]) > 0:
            ax.plot(mms4_mlt[1][mms4_peaks_tind[0]+mms4_tind[0]]*mlt_to_rad, mms4_l[1][mms4_peaks_tind[0]+mms4_tind[0]], '.', color='orange', markersize=10)

        # THEMIS
        if tha_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'

        ax.plot(tha_mlt[1][tha_tind]*mlt_to_rad, tha_l[1][tha_tind], color=color)
        ax.plot(tha_mlt[1][tha_tind][0]*mlt_to_rad, tha_l[1][tha_tind][0], 'o', color=color)
        ax.plot(tha_mlt[1][tha_tind][-1]*mlt_to_rad, tha_l[1][tha_tind][-1], '*', markersize=15, color=color)
        if tha_l[1][tha_tind][0] < 12: 
            ax.text(tha_mlt[1][tha_tind][0]*mlt_to_rad+0.1, tha_l[1][tha_tind][0]-0.5, 'TH-A', fontsize='x-large', color='k')
        elif np.any(tha_l[1][tha_tind] < 12):
            ax.text(tha_mlt[1][tha_tind][np.where(tha_l[1][tha_tind] < 12)[0][0]]*mlt_to_rad, tha_l[1][tha_tind][np.where(tha_l[1][tha_tind] < 12)[0][0]], 'TH-A', fontsize='x-large', color='k')

        if len(tha_peaks_tind[0]) > 0:
            ax.plot(tha_mlt[1][tha_peaks_tind[0]+tha_tind[0]]*mlt_to_rad, tha_l[1][tha_peaks_tind[0]+tha_tind[0]], '.', color='orange', markersize=10)

        if thd_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'

        ax.plot(thd_mlt[1][thd_tind]*mlt_to_rad, thd_l[1][thd_tind], color=color)
        ax.plot(thd_mlt[1][thd_tind][0]*mlt_to_rad, thd_l[1][thd_tind][0], 'o', color=color)
        ax.plot(thd_mlt[1][thd_tind][-1]*mlt_to_rad, thd_l[1][thd_tind][-1], '*', markersize=15, color=color)
        if thd_l[1][thd_tind][0] < 12: 
            ax.text(thd_mlt[1][thd_tind][0]*mlt_to_rad+0.1, thd_l[1][thd_tind][0]-0.5, 'TH-D', fontsize='x-large', color='k')
        elif np.any(thd_l[1][thd_tind] < 12):
            ax.text(thd_mlt[1][thd_tind][np.where(thd_l[1][thd_tind] < 12)[0][0]]*mlt_to_rad, thd_l[1][thd_tind][np.where(thd_l[1][thd_tind] < 12)[0][0]], 'TH-D', fontsize='x-large', color='k')

        if len(thd_peaks_tind[0]) > 0:
            ax.plot(thd_mlt[1][thd_peaks_tind[0]+thd_tind[0]]*mlt_to_rad, thd_l[1][thd_peaks_tind[0]+thd_tind[0]], '.', color='orange', markersize=10)

        if the_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'

        ax.plot(the_mlt[1][the_tind]*mlt_to_rad, the_l[1][the_tind], color=color)
        ax.plot(the_mlt[1][the_tind][0]*mlt_to_rad, the_l[1][the_tind][0], 'o', color=color)
        ax.plot(the_mlt[1][the_tind][-1]*mlt_to_rad, the_l[1][the_tind][-1], '*', markersize=15, color=color)
        if the_l[1][the_tind][0] < 12: 
            ax.text(the_mlt[1][the_tind][0]*mlt_to_rad+0.1, the_l[1][the_tind][0]-0.5, 'TH-E', fontsize='x-large', color='k')
        elif np.any(the_l[1][the_tind] < 12):
            ax.text(the_mlt[1][the_tind][np.where(the_l[1][the_tind] < 12)[0][0]]*mlt_to_rad, the_l[1][the_tind][np.where(the_l[1][the_tind] < 12)[0][0]], 'TH-E', fontsize='x-large', color='k')

        if len(the_peaks_tind[0]) > 0:
            ax.plot(the_mlt[1][the_peaks_tind[0]+the_tind[0]]*mlt_to_rad, the_l[1][the_peaks_tind[0]+the_tind[0]], '.', color='orange', markersize=10)

        # GOES
        if g15_n_peaks == 0: color = 'lightgray'
        else: color = 'gray'

        if len(g15_l_new[1][g15_mltind][np.isfinite(g15_l_new[1][g15_mltind])]) < len(g15_mlt_new[1][g15_mltind][np.isfinite(g15_mlt_new[1][g15_mltind])]):
            ref_arr = g15_l_new[1][g15_mltind]
        else: ref_arr = g15_mlt_new[1][g15_mltind]
        ax.plot(g15_mlt_new[1][g15_mltind][np.isfinite(ref_arr)]*mlt_to_rad, g15_l_new[1][g15_mltind][np.isfinite(ref_arr)], color=color)
        ax.plot(g15_mlt_new[1][g15_mltind][np.isfinite(ref_arr)][0]*mlt_to_rad, g15_l_new[1][g15_mltind][np.isfinite(ref_arr)][0], 'o', color=color)
        ax.plot(g15_mlt_new[1][g15_mltind][np.isfinite(ref_arr)][-1]*mlt_to_rad, g15_l_new[1][g15_mltind][np.isfinite(ref_arr)][-1], '*', markersize=15, color=color)
        if g15_l_new[1][g15_mltind][np.isfinite(ref_arr)][0] < 12: 
            ax.text(g15_mlt_new[1][g15_mltind][np.isfinite(ref_arr)][0]*mlt_to_rad+0.1, g15_l_new[1][g15_mltind][np.isfinite(ref_arr)][0]-0.5, 'GOES-15', fontsize='x-large', color='k')
        elif np.any(g15_l[1][g15_tind] < 12):
            ax.text(g15_mlt[1][g15_mltind][np.where(g15_l[1][g15_mltind] < 12)[0][0]]*mlt_to_rad, g15_l[1][g15_mltind][np.where(g15_l[1][g15_mltind] < 12)[0][0]], 'GOES-15', fontsize='x-large', color='k')

        if len(g15_peaks_mltind[0]) > 0:
            ax.plot(g15_mlt_new[1][g15_peaks_mltind[0]+g15_mltind[0]]*mlt_to_rad, g15_l_mltind[1][g15_peaks_mltind[0]+g15_mltind[0]], '.', color='orange', markersize=10)


        # Earth
        r = np.linspace(0,1,100)
        theta = np.linspace(0,2*np.pi,100)
        R, Theta = np.meshgrid(r,theta)
        Z_black = np.where(((Theta <= np.pi/2) | (Theta >= 3*np.pi/2)), 1, np.nan)
        Z_white = np.where((Theta > np.pi/2) & (Theta < 3*np.pi/2), 1, np.nan)
        ax.contourf(Theta, R, Z_black, levels=[0,1], colors='k', zorder=3)
        ax.contourf(Theta, R, Z_white, levels=[0,1], colors='w', zorder=2)
        ax.contour(Theta, R, R, levels=[0.99], colors='k', linewidths=2)

        # Plot titles
        plt.title('Satellite Locations (in L-Shell and MLT Coordinates); ' + storm_day+' - '+storm_phase)
        plt.savefig(file_path+storm_day+'_'+storm_phase+'_orbits in L-MLT with EMIC.png', bbox_inches='tight', pad_inches=0.25)
        plt.show()
        plt.close()
        
