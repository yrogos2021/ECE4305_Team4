# ------------------------------------------
# Authors: Yossef Naim, Ryan Hennigan, Yael Rogoszinski
# Institution: Worcester Polytechinc Institute
# Date: 02/20/2021
#
# Functionality ... ...
# This python code decodes a BLE (Bluetooth Low Energy) signals/ packets 
# in an offline manner (and potentially in real time) in order to 
# extract the human readable name of the transmitting device as well
# as the hexidecimal data that accompanies it.

import numpy as np
import adi
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
import scipy.integrate as integrate
import scipy.special as special
from scipy.fft import fftshift, fft
from array import *

sample_rate = 61.25e6
center_freq = 2.45e9
exponent = 11
num_samples = 2**exponent

sdr = adi.Pluto("ip:192.168.2.1")

sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_lo = int(center_freq)
sdr.rx_buffer_size = num_samples

T = np.zeros((num_samples, sdr.rx_buffer_size), dtype=float)

for x in range(5):
    for y in range(num_samples):
        data = sdr.rx()
        #print(data)
        T[y,:] = data

# ----------------- Set Fc ------------------
    length = int(len(data)/2)
    half_data = [0] * length
    for i in range(length):
        half_data[i] = data[i]
        

    shifted_f_c_array = fftshift(fft(half_data))
    shifted_f_c_arrayO = fftshift(fft(data))
    f_c = integrate.simps(shifted_f_c_array)
    print(f_c)
    plt.draw()
    plt.pause(0.001)
    time.sleep(0.001)
    plt.yscale('log')
    plt.figure(0)
    plt.plot(abs(shifted_f_c_array))
    plt.yscale('log')
    plt.figure(1)
    plt.plot(abs(shifted_f_c_arrayO))
    
    
plt.show()

# ----------- Stuff From Chpt. 6 ------------

# ------------------ DPLL -------------------

# --------------- Frame Sync ----------------
