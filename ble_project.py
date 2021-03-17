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

sample_rate = 2.00e6

#----------------------- Important Variables ----------------------------
preamble = "0011001100110011"
#preamble = "00001111000011110000111100001111"
#preamble = "000000111111000000111111000000111111000000111111"

access_address = "01101011011111011001000101110001" 
#access_address ="0011110011001111001111111111001111000011000000110011111100000011"

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

exponent = 10
N = 2**exponent
deltaF = 0.0
t = np.linspace(0.0, (N-1)/(float(sample_rate)), N)
PhaseOffset = 0.0

data_phase = np.angle(data)

			
#time_axis = np.linspace(0, num_samples/sample_rate, len(data))
#neg_freq_detrend_line = np.exp(1j*2*np.pi*-0.5*sample_rate*time_axis)

#DC_centered_data = data*neg_freq_detrend_line

#unwrapped_data_phase = np.unwrap(np.angle(DC_centered_data))

#data_phase_derivative = np.diff(unwrapped_data_phase)
		
#for p in potential_packets:
#	print(p + "\n\n")

has_access = []

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
correction_output = np.empty(len(center_data), dtype = complex)
e = np.empty(len(center_data), dtype = complex)
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

bits = "".join(list(map(lambda x: "1" if x < 0 else "0",
		data_phase_derivative)))

potential_packets = bits.split(preamble)

for packet in potential_packets:
	even_bits = packet[::2]
	odd_bits = packet[1::2]
	if even_bits.startswith(access_address):
		has_access = has_access + [even_bits]
	if odd_bits.startswith(access_address):
		has_access = has_access + [odd_bits]

#for p in has_access: 
#	print(p[:1024] + "\n\n")

print("Potential Packets Found: " + str(len(potential_packets)-1))

#dewhiten broadcast bits
dewhittened_packets = []
for packet in has_access:
	dewhittened = dewhiten_str_to_bits(packet[len(access_address):])
	length_field_header = dewhittened[8:14]
	
	payload = dewhittened[16:]
	#payload = payload[::-1]
	#print(payload)
	payload2 = flip(payload)
	print(payload)
	print(payload2)
	payload_hex = hex(int(listToString(payload2), 2))
	payload_hex = payload_hex[2:]
	print(payload_hex)
	#b_payload = bytes.fromhex(payload_hex).decode('utf-8')
	#payload_final = b_payload.decode("ASCII")
	#print(" ")
	#print("Payload: ", payload_final)
	
	'''
	length_device_name = dewhittened[20:22]
	payload_size_bytes = int("".join(str(x) for x in length_field_header[::-1]), 2)
	name_length = int("".join(str(x) for x in length_device_name[::-1]), 2)
	print("Found payload size (bytes): " + str(payload_size_bytes))
	total_length_after_AA = 16 + 8*payload_size_bytes + 24
	chopped_packet = dewhittened[:total_length_after_AA]
	
	
	p = listToString(chopped_packet[27:27+name_length])
	#"".join(str(x) for x in chopped_packet)#
	print("Name size: ", name_length)
	
	chopped_hex = hex(int(listToString(chopped_packet)))
	chopped_hex = chopped_hex[2:]
	hex_backs = hex(int(listToString(chopped_packet[::-1])))
	name = chopped_hex[27:27+(name_length*2)]
		
	
	print("Total packet: ", listToString(chopped_packet))
	print("Hex Format: ", chopped_hex)
	print("Hex Format Reverse: ", hex_backs)
	print("Name: ", name)#p_slice)#[11:11+name_Size])
	#p_bytes = bytes.fromhex(p_slice).decode('utf-8')
	#p_string = p_bytes.decode("ASCII")
	#print("Translation: ", p_string)
	
	#decode the packet information
#	if(len(potential_packets) - 1) == 37:
		'''
	
#print(dewhittened_packets)

#print(has_access)
plt.plot(data_phase_derivative)
plt.show()


