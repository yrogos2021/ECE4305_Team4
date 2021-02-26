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

t = np.zeros((num_samples, sdr.rx_buffer_size), dtype=float)

for x in range(5):
    for y in range(num_samples):
        data = sdr.rx()
        #print(data)
        t[y,:] = data

    shifted_spectrum = fftshift(fft(data))
    plt.draw()
    plt.pause(0.001)
    time.sleep(0.001)
    plt.figure(0)
    plt.plot(abs(shifted_spectrum))
    plt.yscale('log')
    plt.figure(1)
    T = np.abs(t)
    dx, dy = 0.015, 0.05
    y, x = np.mgrid[slice(-4, 4 + dy, dy),
                slice(-4, 4 + dx, dx)]

    T_min, T_max = -np.abs(T).max(), np.abs(T).max()
    plt.imshow(T, cmap='PuRd', vmin = T_min, vmax = T_max,
                extent = [x.min(), x.max(), y.min(), y.max()],
                interpolation = 'nearest', origin = 'lower')
    

plt.show()
