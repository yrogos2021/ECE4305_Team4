## ------------------------------------------
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
from scipy.fft import fftshift, fft

# --------- Radio Configuration ---------

# Center and Sampling Frequencies
sample_rate = 8e6
symbol_rate = 1e6
center_freq = 2.45e9
exponent = 10
N = 2**exponent

# More Setup
sdr = adi.Pluto ("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_lo = int(center_freq)
sdr.rx_buffer_size = 100
samplesRecieved = 0

data = sdr.rx()

# --------- Constants ---------
deltaF = 0.0
t = np.linspace(0.0, (N-1)/(float(sample_rate)), N)
PhaseOffset = 0.0

data_phase = np.angle(data)

# --------- Ideal Signal ---------
offset = 1e6
Ideal_dataI = np.cos(2.0*np.pi*(offset+deltaF)*t+PhaseOffset*np.ones(N))
Ideal_dataQ = -np.sin(2.0*np.pi*(offset+deltaF)*t+PhaseOffset*np.ones(N))

# --------- Time to Frequency and Shifted ---------
time_to_freq = fft(data)
shifted_frequency = fftshift(time_to_freq)
energy_sum = sum(shifted_frequency)

# ----------------- Set Fc ------------------
half_energy = energy_sum/2
sample_center = 0
summation = 0
for i in range(N):
	summation = summation + shifted_frequency[i]
	if summation < half_energy:
		pass
	else:
		sample_center = i
		break

bin_w = sample_rate/len(shifted_frequency)
num_offset = sample_center - len(shifted_frequency)/2
foffset = -bin_w * num_offset

# ----------- Coarse Frequency Correction ------------
time_axis = np.linspace(0, N/sample_rate, len(data))
neg_freq_detrend_line = np.exp(1j*2*np.pi*0.5*sample_rate*time_axis)
coarse_freq_correct = np.exp(1j*2*np.pi*foffset*time_axis)

# ----------- Coarse Frequency Correction ------------
center_data = data*neg_freq_detrend_line*coarse_freq_correct
DC_centered_data = data*neg_freq_detrend_line

unwrapped_data_phase_DC = np.unwrap(np.angle(DC_centered_data))
unwrapped_data_phase = np.unwrap(np.angle(center_data))
#unwrapped_compensator_phase = np.unwrap(compensator_phase)

data_phase_derivative = np.diff(unwrapped_data_phase)

shifted = data_phase_derivative**2
shifted = list(map(lambda x: 0 if x < 4 else x, shifted))

# ------------------ DPLL -------------------
#for G
B_L = 0.01
damping = 1
M = 2e6
K = 1
theta = B_L/(M*((damping+0.25)/damping))
delta = 1+ (2*damping *theta)+ (theta**2)
G = (4*damping *theta/delta)/(M*K)

#for DPLL
correction_output = np.empty(N, dtype = complex)
e = np.empty(N, dtype = complex)
f_n = np.zeros(N)
loop_filter= np.empty(N, dtype=complex)
loop_filter_past = 0.0
e_past = 0.0

for i in range(len(center_data)):
	#phase rotate
	if i==0:
		phase_rotator = center_data[i] * deltaF
		correction_output[i] = phase_rotator
	else:
		phase_rotator = center_data[i] * new_delta
		correction_output[i] = phase_rotator	
	# Errorf
	e[i] = phase_rotator * (Ideal_dataI[i] +1j*Ideal_dataQ[i])
	
	loop_filter[i] = loop_filter_past + G * e_past
	e_past = e[i]
	loop_filter_past = loop_filter[i]

	# DDS - this is if we use the summation format

	f_n_angle = np.angle(loop_filter[i])
	new_delta= np.exp(-1j*f_n_angle)
	
# --------------- Frame Sync ----------------


#----------------------------------------------------

