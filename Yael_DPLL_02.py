import time

import adi
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Create radio
sdr = adi.Pluto()

sample_rate = 8e6
symbol_rate = 1e6
center_freq = 2.45e9
exponent = 10
N = 2**exponent

sdr = adi.Pluto ("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_lo = int(center_freq)
sdr.rx_buffer_size = N

data = sdr.rx()
foffset = 1.0e6 # integral added here

#Phase_Offset = 0.0

#t = np.linspace(0.0, (N-1)/(float(sample_rate)),N)
t = N*(1/sample_rate)
#t = np.linspace(0.0,(N-1)/(float(sample_rate)),N)  

deltaF = 0.0
phi = np.angle(data)

#for loop from 0 to N-1
e = np.empty(N, dtype=complex)
f_n = np.zeros(N)
#for G
B_L = 0.01
damping = 1
M = 2e6
K = 1
theta = B_L/(M*((damping+0.25)/damping))
delta = 1+ (2*damping *theta)+ (theta**2)
G = (4*damping *theta/delta)/(M*K)
loop_filter= np.empty(N, dtype=complex)
loop_filter_past = 0.0
e_past = 0.0

data_phase = np.angle(data)

#DPLL
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

#fiter_derivative = np.diff(np.angle(f_n))	

#dataI = np.real(e)
#dataI = np.cos(2.0*np.pi*(Foffset+deltaF)*t+PhaseOffset*np.ones(N)) # Inphase data samples
dataI = np.cos(2.0*np.pi*(foffset+deltaF)*t+e*np.ones(N)) # Inphase data samples
#dataQ = np.imag(e)
#dataQ = -np.sin(2.0*np.pi*(Foffset+deltaF)*t+PhaseOffset*np.ones(N)) # Quadrature data samples
dataQ = -np.sin(2.0*np.pi*(foffset+deltaF)*t+e*np.ones(N)) # Quadrature data samples


# Plot signal constellation diagram
plt.figure(figsize=(9, 5))
plt.plot(dataI,dataQ)
plt.xlabel('Inphase')
plt.ylabel('Quadrature');
plt.show()