import numpy as np
import matplotlib.pyplot as plt

# Load the image and extract a row/column
img = plt.imread("dogsmall.jpg")
signal = img[0, :]  # extract the first row


# Compute the DFT using the naive method
N = len(signal)
dft = np.zeros(N, dtype=np.complex128)
for k in range(N):
    for n in range(N):
        dft[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)

# Perform the filtering operation by attenuating one component
dft[3] /= 2  # attenuate the 4th component (remember that indexing starts at 0)

# Compute the IDFT to get the filtered signal
filtered_signal = np.zeros(N, dtype=np.float64)
for n in range(N):
    for k in range(N):
        filtered_signal[n] += dft[k] * np.exp(2j * np.pi * k * n / N).real / N

# Compute the FFT using NumPy
fft = np.fft.fft(signal)

# Plot the results side by side
plt.subplot(1, 3, 1)
plt.plot(signal)
plt.title("Original Signal")

plt.subplot(1, 3, 2)
plt.plot(filtered_signal)
plt.title("Filtered Signal (DFT)")

plt.subplot(1, 3, 3)
plt.plot(np.real(fft))
plt.title("Filtered Signal (FFT)")

plt.show()
