# ------------------------------------------
# Author: Yossef Naim
# Institution: Worcester Polytechinc Institute
# Date: 02/20/2021
#
# Functionality ... ...
# This python code decodes a BLE (Bluetooth Low Energy) signals/ packets
# in an offline manner (and potentially in real time) in order to
# extract the human readable name of the transmitting device as well
# as the hexidecimal data that accompanies it.

from array import *
import adi
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special

# Definitions
#def integrand(x,a,b)
#    return


# Center and Sampling Frequencies
sample_rate = 8e6
symbol_rate = 1e6
center_freq = 2.45e9
exponent = 10
num_samples = 2**exponent

# More Setup
sdr = adi.Pluto ("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_lo = int(center_freq)
sdr.rx_buffer_size = 100
samplesRecieved = 0

data = sdr.rx()

# --------- Constants ---------
#t = np.linspace(0.0, (N-1)/(float(sample_rate)),N)
t = N*(1/sample_rate)

deltaF = 0.0
phi = angle(data)

t = N*(1/sample_rate)

deltaF = 0.0
phi = angle(data)

#for loop from 0 to N-1
e = []
f_n = np.zeros(N)
#for G
B_L = 0.01
damping = 1
M = 2e6
K = 1
theta = B_L/(M*((damping+0.25)/damping))
delta = 1+ (2*dampng *theta)+ (theta**2)
G = (4*damping *theta/delta)/(M*K)
loop_filter= np.empty(N, dtype=complex)
loop_filter_past = 0.0
e_past = 0.0

# --------- Take Spectrum of Signal ---------
time_to_freq = numpy.fft(data)
shifted_frequency = numpy.fft.fftshift(time_to_freq)
energy_sum = sum(shifted_frequency)

# ----------------- Set Fc ------------------
half_energy = energy_sum/2
sample_center = 0
summation = 0
for i in range(num_samples):
	summation = summation + shifted_frequency[i]
	if summation < half_energy:
		pass
	else:
		sample_center = i
		break

bin_w = sample_rate/len(shifted_frequency)
num_offset = sample_center - len(shifted_frequency)/2
foffset = -bin_w * num_offset

# ----------- Stuff From Chpt. 6 ------------
time_axis = np.linspace( 0, num_samples/sample_rate, len(data))
neg_freq_detrend_line = np.exp(1j*2*np.pi.0.5*sample_rate*time_axis)
coarse_freq_correct = np.exp(1j*2*np.pi*foffset*time_axis)
center_data = data*neg)freq_detrend_line*coarse_freq_correct

unwrapped_data_phase = np.unwrap(np.angle(center_data)
unwrapped_compensator_phase = np.wrap(compensator_phase)

data_phase_derivative = np.diff(unwrapped_data_phase)

shifted = data_phase_derivative**2
shifted = list(map(lambda x: 0 if x < 4 else x, shifted))

# ------------------ DPLL -------------------

for i in range(N):
	#phase rotate
	if i==0:
		gamma = data_phase - (2*np.pi*deltaF*t)
	else:
		gamma = data_phase - (2*np.pi*new_delta*t)

	# Error
	if np.all(abs(2*np.pi*foffset*t - gamma[i]) > abs(2*np.pi*-foffset*t - gamma[i])):
		e[i] = 2*np.pi*(-foffset*t) - gamma[i] # '0'
	else:
		e[i] = 2*np.pi*foffset*t - gamma[i] # '1'
	loop_filter[i] = loop_filter_past + G * e_past
	e_past = e[i]
	loop_filter_past = loop_filter[i]

	# DDS - this is if we use the summation format

	f_n_angle = np.angle(f_n[i])
	new_delta= np.exp(-1j*f_n_angle)
	
# --------------- Frame Sync ----------------


----------------------------------------------------
