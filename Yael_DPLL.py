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
sdr.rx_buffer_size = num_samples

data = sdr.rx()
foffset = 1.0e6 # integral added here

#Phase_Offset = 0.0

#t = np.linspace(0.0, (N-1)/(float(sample_rate)),N)
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

# Error
if (abs(2*np.pi*foffset*t - gamma) > abs(2*np.pi*-foffset*t - gamma)):
	e[0] = 2*np.pi*(-foffset*t) - gamma) # '0'
else:
	e[0] = 2*np.pi*foffset*t - gamma) # '1'

#DPLL
for i in range(N):
	#phase rotate
	gamma = data - (2*np.pi*deltaF*t)

	# Error
	if (abs(2*np.pi*foffset*t - gamma) > abs(2*np.pi*-foffset*t - gamma)):
		e[i] = 2*np.pi*(-foffset*t) - gamma) # '0'
	else:
		e[i] = 2*np.pi*foffset*t - gamma) # '1'

	#Loop Filter Using Z-Transform
	#if(i == 0):
	#	f_n[0] = G*e[0]	
	#else:
	#	f_n[i] = f_n[i-1] + G*e[i-1]
		
	f_n[N] = f_n[0] + G*(sum(e[i])

	
return #something		

##plt.figure(figsize=(9, 5))
#plt.plot(dataI, dataQ)
#xlabel('Inphase')
#plt.ylabel('Quadrature')
#plt.show()

#mag = np.sqrt(np.square(np.real(data))+np.square(np.imag(data)))
#phase = np.arctan(np.imag(data)/np.real(data))
#dataI = np.cos(2.0*np.pi*(foffset+deltaF)*t+Phase_Offset*np.ones(N))
#dataQ = -np.sin(2.0*np.pi*(foffset+deltaF)*t+Phase_Offset*np.ones(N))
