"""SYNTHETIC stand-in for SNLS80mV.mat - ONLY for smoke-testing the pipeline.

Not the real Silverbox data.  Duffing oscillator  y'' + a y' + b y + c y^3 = g u
driven by a random-phase multisine (0-200 Hz, 80 mV rms) that starts at sample
40650, sampled at 610.35 Hz, 131072 samples, small output noise.
Saved with variables V1 (input) and V2 (output) as 1 x N row vectors.
"""
import sys

import numpy as np
import scipy.io as sio

FS = 610.35
N = 131072
START = 40650
P = 8192            
OS = 8              

a, b, c, g = 45.0, (2 * np.pi * 68.0) ** 2, 4.0e5, 6.0e4
noise = 1.0e-3

out = sys.argv[1] if len(sys.argv) > 1 else "SNLS80mV_synthetic.mat"
rng = np.random.default_rng(1)

K = int(200.0 * P / FS)
A = 0.08 * np.sqrt(2.0 / K)
n_up = OS * P
X = np.zeros(n_up // 2 + 1, dtype=complex)
X[1:K + 1] = (n_up / 2.0) * A * np.exp(1j * rng.uniform(0, 2 * np.pi, K))
period = np.fft.irfft(X, n=n_up)                      
u_fine = np.tile(period, N // P + 1)[: N * OS].copy()
u_fine[: START * OS] = 0.0                            

h = 2.0 / (OS * FS)                                  
y = v = 0.0
ys = np.empty(N)
ys[0] = 0.0
f = lambda y, v, u: (v, g * u - a * v - b * y - c * y ** 3)
for n in range(1, N):
    base = (n - 1) * OS
    for j in range(OS // 2):
        i = base + 2 * j
        u0, um, u1 = u_fine[i], u_fine[i + 1], u_fine[i + 2]
        k1y, k1v = f(y, v, u0)
        k2y, k2v = f(y + 0.5 * h * k1y, v + 0.5 * h * k1v, um)
        k3y, k3v = f(y + 0.5 * h * k2y, v + 0.5 * h * k2v, um)
        k4y, k4v = f(y + h * k3y, v + h * k3v, u1)
        y += h / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
        v += h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    ys[n] = y

u = u_fine[::OS]
yn = ys + noise * rng.standard_normal(N)
sio.savemat(out, {"V1": u[None, :], "V2": yn[None, :]})

s = slice(42650, 46746)
print(f"wrote {out}: N={N}")
print(f"excerpt: u rms {u[s].std():.4f} V, y rms {yn[s].std():.4f} V, y peak {np.abs(yn[s]).max():.3f} V")
print(f"test-record output RMS (zero baseline) {np.sqrt(np.mean((yn[s][3072:] - yn[s].mean())**2)):.4f} V")
