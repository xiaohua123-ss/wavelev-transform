import pywt
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 1, num=2048)
signal = np.sin(250 * np.pi * x ** 2)



coeffs = pywt.wavedec(signal, 'db2', level=8)
coeffs[1:] = (pywt.threshold(i, value=0.2, mode="soft" ) for i in coeffs[1:])
reconstructed_signal = pywt.waverec(coeffs, 'db2')
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(signal[:1000], label='signal')
ax.plot(reconstructed_signal[:1000], label='reconstructed signal', linestyle='--')
ax.legend(loc='upper left')
ax.set_title('de- and reconstruction using wavedec()')
plt.show()