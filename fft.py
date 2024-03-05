
import numpy as np

def fft(x):
    n = len(x)
    if n == 1:
        return x
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])

        factor = np.exp(-2j*np.pi*np.arange(n)/n)

        X = np.concatenate([X_even + factor[:int(n/2)]*X_odd, X_even + factor[int(n/2):]*X_odd])
    return X

print(np.ndarray.round(fft([1, 1, 1, 1])))