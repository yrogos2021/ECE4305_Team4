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
	
	for x in s:
		character = str(x)
		str1 += character
	return str1
	
#----------------------- Important Variables ----------------------------
sample_rate = 2.00e6
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
time_axis = np.linspace(0, num_samples/sample_rate, len(data))
neg_freq_detrend_line = np.exp(1j*2*np.pi*-0.5*sample_rate*time_axis)

DC_centered_data = data*neg_freq_detrend_line

unwrapped_data_phase = np.unwrap(np.angle(DC_centered_data))

data_phase_derivative = np.diff(unwrapped_data_phase)

bits = "".join(list(map(lambda x: "1" if x < 0 else "0",
		data_phase_derivative)))
		
potential_packets = bits.split(preamble)
#for p in potential_packets:
#	print(p + "\n\n")

has_access = []

for packet in potential_packets:
	even_bits = packet[::2]
	odd_bits = packet[1::2]
	if even_bits.startswith(access_address):
		has_access = has_access + [even_bits]
	if odd_bits.startswith(access_address):
		has_access = has_access + [odd_bits]

for p in has_access: 
	print(p[:1024] + "\n\n")

print("Potential Packets Found: " + str(len(potential_packets)-1))

#dewhiten broadcast bits
dewhittened_packets = []
for packet in has_access:
	dewhittened = dewhiten_str_to_bits(packet[len(access_address):])
	length_field_header = dewhittened[8:14]
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
	hex_backs = chopped_hex[::-1]
	name = chopped_hex[27:27+(name_length*2)]
	
	#p_slice = hex(int(p, 2))
	#p_slice = p_slice[2:]
	#p_slice = p_slice[::-1]
	
	
	print("Total packet: ", listToString(chopped_packet))
	print("Hex Format: ", chopped_hex)
	print("Hex Format Reverse: ", hex_backs)
	print("Name: ", name)#p_slice)#[11:11+name_Size])
	#p_bytes = bytes.fromhex(p_slice).decode('utf-8')
	#p_string = p_bytes.decode("ASCII")
	#print("Translation: ", p_string)
	
	#decode the packet information
#	if(len(potential_packets) - 1) == 37:
		
	
print(dewhittened_packets)

#print(has_access)
plt.plot(data_phase_derivative)
plt.show()
