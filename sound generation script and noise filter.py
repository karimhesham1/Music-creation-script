from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

N = 3*1024

f = np. linspace(0 , 512 , int(N/2))
t = np. linspace(0 , 3 , 12*1024)

f1, f2 = np. random. randint(0, 512, 2)

noise = np.sin(2*f1*np.pi*t) + np.sin(2*f2*np.pi*t) * (t >= 0) * (t <= 3)

F = [369.99,293.66,293.66,329.63,349.23,329.63,293.66,277.18,293.66,329.63,369.99,369.99,493.88,277.18]

T = [0.3,0.3,0.1,0.1,0.1,0.15,0.15,0.15,0.25,0.25,0.2,0.1,0.2]
ti = [0,0.5,1,1.1,1.2,1.4,1.6,1.8,2,2.25,2.5,2.7,2.8,2.9]

i = 0
x = 0
while i < (len(F)-1) :
    x = x + np.reshape(np.sin(2*np.pi*F[i]*t)*[(t>=ti[i])&(t<=(T[i]+ti[i]))],np.shape(t))
    i += 1

x_f = fft(x)
x_f = 2/N * np.abs(x_f [0:np.int(N/2)])

x_noise = x + noise

x_noise_f = fft(x_noise)
x_noise_f = 2/N * np.abs(x_noise_f [0:np.int(N/2)])

ind = np.argpartition(x_noise_f, -2)[-2:]

fn11 = np.round(f[ind[0]])
fn22 = np.round(f[ind[1]])

x_filtered = x_noise - (np.sin(2*fn11*np.pi* t) + np.sin(2*fn22*np.pi* t)) * (t >= 0) * (t <= 3)

x_filtered_f = fft(x_filtered)
x_filtered_f = 2/N * np.abs(x_filtered_f [0:np.int(N/2)])




plt.subplot(3,2,1)
plt.plot(t,x)
plt.subplot(3,2,2)
plt.plot(f,x_f)

plt.subplot(3,2,3)
plt.plot(t,x_noise)
plt.subplot(3,2,4)
plt.plot(f,x_noise_f)

plt.subplot(3,2,5)
plt.plot(t,x_filtered)
plt.subplot(3,2,6)
plt.plot(f,x_filtered_f)


sd.play(x_filtered, 3 * 1024)
