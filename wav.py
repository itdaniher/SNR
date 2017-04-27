import numpy as np
import scipy.io.wavfile

s_rate = 48e3

noises = [10**i for i in range(5)]

t = np.linspace(0, 1, s_rate)

freq = 1200
#import matplotlib.pyplot as plt

for mag in noises:
    x = (np.random.rand(512)-0.5)*mag
    x += np.sin(freq*t[:512]*2*np.pi)
    #plt.plot(np.abs(np.fft.fft(x)))
    #plt.figure()
    scipy.io.wavfile.write(str(mag)+"+1200hz.wav", int(s_rate), x)

#plt.show()
