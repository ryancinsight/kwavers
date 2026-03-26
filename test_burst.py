import traceback
import sys
import numpy as np

try:
    from kwave.utils.signals import tone_burst
    import pykwavers as kwa
    
    sf = 10e6
    cf = 1e6
    cyc = 3
    
    k = tone_burst(sf, cf, cyc)
    q = kwa.tone_burst(sf, cf, cyc)
    
    print(f"kw shape: {k.shape} kwa shape: {q.shape}")
    k = np.squeeze(k)
    q = np.squeeze(q)
    
    if len(k) > len(q):
        k = k[:len(q)]
    elif len(q) > len(k):
        q = q[:len(k)]
        
    diff = np.abs(k - q)
    print(f"Max diff: {np.max(diff)}")
    idx = np.argmax(diff)
    
    # Print values around max diff
    start = max(0, idx - 2)
    end = min(len(k), idx + 3)
    
    print("k-wave: ", k[start:end])
    print("kwavers:", q[start:end])
    print("diff:   ", diff[start:end])
    
except Exception as e:
    traceback.print_exc()
