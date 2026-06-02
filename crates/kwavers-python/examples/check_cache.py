import numpy as np

data = np.load("kw_results.npz")
print(f"pressure shape: {data['pressure'].shape}")
print(f"pressure max: {np.max(data['pressure']):.4e}")
print(f"pressure min: {np.min(data['pressure']):.4e}")
print(f"dt: {float(data['dt']):.6e}")
print(f"input_signal shape: {data['input_signal'].shape}")
print(f"input_signal max: {np.max(np.abs(data['input_signal'])):.4e}")
