#imports
import numpy as np
import adi
import time
import matplotlib.pyplot as plt
from scipy.fft import fftshift, fft
from scipy import signal

#configure pluto
sample_rate = 8e6
center_freq = 2.45e9
exponent = 10
num_samples = 2**exponent

sdr = adi.Pluto ("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_lo = int(center_freq)
sdr.rx_buffer_size = num_samples

data = sdr.rx()

x = np.pi/4
while (x != 0):
    # Get mag/phase representation of samples
    mag = np.sqrt(np.square(np.real(data))+np.square(np.imag(data)))
    phase = np.arctan(np.imag(data)/np.real(data))
    st = mag*phase
    #print(st)

    # Multiply by phase rotator
    rotator = np.exp(-1j*x)
    rt = st*rotator
    #print(rt)

    # Calculate phase error
    new_phase = np.angle(rt)
    error = phase-new_phase
    #print(error)

    # Run error through loop filter (LPF)
    order = 1
    cutoff = 200
    normalized_cutoff = 2*cutoff/sample_rate
    b,a = signal.butter(order,normalized_cutoff,'low')
    smooth_error = signal.lfilter(b,a,error)
    #print(smooth_error)

    # Estimate new phase
    x = np.average(smooth_error)


    plt.plot(error)
    plt.plot(x)
    plt.plot(smooth_error)
    plt.show()