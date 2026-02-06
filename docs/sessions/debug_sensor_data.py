import numpy as np

# Load the sensor data
data = np.load('examples/results/sensor_data.npz')

print("Available keys:", list(data.keys()))
print()

for key in data.keys():
    arr = data[key]
    print(f"{key}:")
    print(f"  Shape: {arr.shape}")
    print(f"  Min: {arr.min():.6e}")
    print(f"  Max: {arr.max():.6e}")
    print(f"  Mean: {arr.mean():.6e}")
    print(f"  Std: {arr.std():.6e}")
    
    # Show first 10 samples
    if arr.ndim > 0 and len(arr) >= 10:
        print(f"  First 10 samples: {arr[:10]}")
    print()
