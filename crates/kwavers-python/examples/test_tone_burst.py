import numpy as np
from kwave.utils.signals import tone_burst
import matplotlib.pyplot as plt

kw_dt = 2e-7
tone_burst_freq = 1e6
tone_burst_cycles = 3

sample_freq = 1 / kw_dt
print(f"Sample freq: {sample_freq}")

input_signal = tone_burst(sample_freq, tone_burst_freq, tone_burst_cycles).flatten()

for i, v in enumerate(input_signal):
    print(f"{i:02d}: {v: >10.4f}")

plt.plot(input_signal, marker='o')
plt.title('tone_burst output')
plt.savefig('tone_burst_test.png')
