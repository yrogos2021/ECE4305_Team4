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
import math
import statistics
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
import scipy.integrate as integrate
import scipy.special as special
from scipy.fft import fftshift, fft
from scipy import signal
from array import *

#--------- Definitions ----------
def str_xor(a,b):
	return list(map(lambda x: 0 if x[0] is x[1] else 1, zip(a,b)))
	
def dewhiten_str_to_bits(bits):
	#Need to figure out initialization on channel other than channel 38
	current_state = [[1,1,0,0], [1,1,0]]
	lfsr_out = ""
	for i in range(len(bits)):
		out_bit = current_state[1][-1]
		lfsr_out = lfsr_out + str(out_bit)
		current_state[1] = [current_state[0][-1] ^ out_bit] + current_state[1][:-1]
		current_state[0] = [out_bit] + current_state[0][:-1]
	return str_xor(bits, lfsr_out)

def listToString(s):
	str1 = ""
	
	for x in range(len(s)):
		character = str(s[x])
		str1 += character
	return str1
'''
test1 = [1,0,1,1,0,0,1,1]
test2 = listToString(test1)
print(test2)
print(int(test2,2))
print(hex(int(test2,2)))
'''
def flip(byte):
	temp0 = []
	size = int((len(byte))/8)
	for x in range(size):
		temp = byte[8*x:8*x+8]
		temp = temp[::-1]
		temp0 += temp
	return temp0
'''
test = [0,0,0,0,1,1,1,1,0,1,0,1,0,0,1,1]
test1 = flip(test)
print(test)
print(test1)
'''

	#byte = byte[::-1]	
#----------------------- Important Variables ----------------------------
sample_rate = 2.00e6
preamble = "0011001100110011"

access_address = "01101011011111011001000101110001" 

center_freq = 2.426e9 #BLE #Recommend: use chnl 38
exponent = 21
num_samples = 2**exponent #apporx 4 million samples
freq_axis = np.linspace(center_freq-0.5*sample_rate,
			center_freq+0.5*sample_rate,
			num_samples)
			
sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_lo = int(center_freq)
sdr.rx_buffer_size = num_samples

data = sdr.rx()
time_axis = np.linspace(0, num_samples/sample_rate, len(data))
neg_freq_detrend_line = np.exp(1j*2*np.pi*-0.5*sample_rate*time_axis)

DC_centered_data = data*neg_freq_detrend_line

unwrapped_data_phase = np.unwrap(np.angle(DC_centered_data))

data_phase_derivative = np.diff(unwrapped_data_phase)

bits = "".join(list(map(lambda x: "1" if x < 0 else "0",
		data_phase_derivative)))
		
potential_packets = bits.split(preamble)
has_access = []

for packet in potential_packets:
	even_bits = packet[::2]
	odd_bits = packet[1::2]
	if even_bits.startswith(access_address):
		has_access = has_access + [even_bits]
	if odd_bits.startswith(access_address):
		has_access = has_access + [odd_bits]


#dewhiten broadcast bits
dewhittened_packets = []
for packet in has_access:
	dewhittened = dewhiten_str_to_bits(packet[len(access_address):])	
	payload = dewhittened[16:]
	payload2 = flip(payload)
	payload3 = int(listToString(payload2),2)
	payload_hex = hex(payload3)
	payload_hex = payload_hex[2:]
	print(payload_hex)
	