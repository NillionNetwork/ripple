import pywt
import numpy as np
import matplotlib.pyplot as plt
import json

# Set parameters
n = 24 # bit width
d = 16 # domain width
f = 12 # precision
J = 8  # wave depth

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_real = np.linspace(-2**(d-1-f), 2**(d-1-f)-2**-f, 2**d)
y_real = sigmoid(x_real)
y_coeffs, *_ = pywt.wavedec(y_real, 'bior2.2', level=J)
print(len(y_coeffs), 2**(d-J))

def real_to_fixed(x, n, f):
    x_fixed = np.floor(np.array(x) * 2**f).astype(int)
    # x_fixed[x_fixed < 0] = (x_fixed + 2**n)[x_fixed < 0]
    return x_fixed

def fixed_to_real(x, n, f):
    x[x >= 2**(n-1)] = (x - 2**n)[x >= 2**(n-1)]
    return x / 2**f

def lut_eval(xl, xs, lut):
    y = np.zeros_like(xl)
    for i, (msb, lsb) in enumerate(zip(xl, xs)):
        # Parseval Inner Product
        r = lut[int(msb) - 2] * (2**J - lsb) + lut[int(msb) - 1] * lsb
        #r = lut[int(msb)] * (2**J - lsb-1) + lut[int(msb)+1] * (lsb+1)
        # Big Truncation
        y[i] = r // 2**(1.5*J)
    return y

x_real = np.linspace(-2**(d-1-f), 2**(d-1-f)-2**-f, 2**d)
y_real = sigmoid(x_real)
y_fixed = real_to_fixed(y_real, n, f)
x_fixed = real_to_fixed(x_real, n, f)
x_trunced = np.floor(x_fixed * 2**(-J))
x_lsb = x_fixed % 2**J

# Normalize Tables
y_coeffs_scaled = y_coeffs

lut_db = real_to_fixed(y_coeffs_scaled, n, f)
lut_db = np.roll(lut_db, 2**(d-J-1))
print(f"{np.max(lut_db) = } {np.min(lut_db) = }")

print('LUT (Db) size:', len(lut_db))
json_object = {str(index): str(value) for index, value in enumerate(lut_db)}
json_string = json.dumps(json_object)
file_path = '../data/bior_lut.json'
with open(file_path, 'w') as file:
    file.write(json_string)


