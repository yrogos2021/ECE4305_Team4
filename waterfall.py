#imports
import numpy as np
import adi
import time
import matplotlib.pyplot as plt
from scipy.fft import fftshift, fft
from scipy import signal

#configure pluto
sample_rate = 60e6
center_freq = 2.45e9
exponent = 10
num_samples = 2**exponent

sdr = adi.Pluto ("ip:192.168.2.1")

sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_lo = int(center_freq)
sdr.rx_buffer_size = num_samples

for x in range(15):
    data = sdr.rx()
    shifted_spectrum = fftshift(fft(data))
    print(shifted_spectrum)
    spec2 = abs(np.square(np.real(shifted_spectrum)))+abs(np.square(np.imag(shifted_spectrum)))
    #plt.figure(0)
    #plt.specgram(abs(shifted_spectrum),scale='dB')
    #plt.figure(1)
    plt.draw()
    plt.pause(0.001)
    time.sleep(0.001)
    plt.specgram(spec2,scale='dB')

print("done")
plt.show()
