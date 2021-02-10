echo "# ECE4305_Team6" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/yrogos2021/ECE4305_Team6.git
git push -u origin main

import numpy as np
import adi
import matplotlib.pyplot as plt
import time

sample_rate = 10e6 # Hz
center_freq = 100e6 # Hz

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.rx_lo = int(center_freq)
sdr.rx_buffer_size = 1024 # this is the buffer the Pluto uses to buffer samples
samples = sdr.rx() # receive samples off Pluto

start_time = time.time()
# do stuff
end_time = time.time()
print('seconds elapsed:', end_time - start_time)
