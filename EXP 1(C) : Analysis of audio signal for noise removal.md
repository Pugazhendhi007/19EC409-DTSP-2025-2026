# EXP 1(C) : Analysis of audio signal for noise removal

# AIM: 

# To analyse an audio signal and remove noise

# APPARATUS REQUIRED:  
PC installed with SCILAB. 

# PROGRAM: 
```
from google.colab import files
uploaded = files.upload()
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample, wiener
from scipy.fftpack import fft, fftfreq
from IPython.display import Audio, display

# 1️⃣ Load audio files
fs_speech, speech = wavfile.read("dtsp_audio.wav")
fs_noise, noise = wavfile.read("my_noise.wav")

# 2️⃣ Convert stereo to mono
if speech.ndim > 1:
    speech = speech.mean(axis=1)
if noise.ndim > 1:
    noise = noise.mean(axis=1)

# 3️⃣ Resample noise if needed
if fs_speech != fs_noise:
    noise = resample(noise, len(speech))
    fs = fs_speech
else:
    fs = fs_speech
    noise = noise[:len(speech)]

# 4️⃣ Mix speech + noise
noisy_signal = speech + 0.5 * noise  # adjust 0.5 to control noise level

# 5️⃣ Plot function
def plot_spectrum(signal, fs, title):
    N = len(signal)
    freq = fftfreq(N, 1/fs)
    magnitude = np.abs(fft(signal))
    plt.figure(figsize=(10,4))
    plt.plot(freq[:N//2], magnitude[:N//2])
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()

# 6️⃣ Plot frequency spectra
plot_spectrum(speech, fs, "Spectrum of Original Speech")
plot_spectrum(noisy_signal, fs, "Spectrum of Noisy Speech")

# 7️⃣ Apply Wiener filter to remove noise
cleaned_signal = wiener(noisy_signal + 1e-8)  # small offset avoids warnings

# 8️⃣ Plot cleaned spectrum
plot_spectrum(cleaned_signal, fs, "Spectrum after Noise Removal")

# 9️⃣ Save cleaned audio
wavfile.write("cleaned_output.wav", fs, cleaned_signal.astype(np.float32))

# 🔊 10️⃣ Play the audios
print("Original Speech:")
display(Audio(speech, rate=fs))

print("Noisy Speech:")
display(Audio(noisy_signal, rate=fs))

print("Cleaned Speech:")
display(Audio(cleaned_signal, rate=fs))

print("✅ Done! 'cleaned_output.wav' saved.")

```
Original Clean Audio 
[good-morning-242169.1 (1).mp3](https://github.com/user-attachments/files/23619595/good-morning-242169.1.1.mp3)
Noise Sample 
[intro-noise-131718.mp3](https://github.com/user-attachments/files/23619614/intro-noise-131718.mp3)

Noisy (Merged) Audio 
[noisy.merged.audio.wav](https://github.com/user-attachments/files/23619629/noisy.merged.audio.wav)

Extracted Noise removed 
[Extracted.Noise.removed.wav](https://github.com/user-attachments/files/23619631/Extracted.Noise.removed.wav)

# OUTPUT:
<img width="1012" height="393" alt="image" src="https://github.com/user-attachments/assets/5e394406-0a3e-450a-bcc1-4ae9293c2af6" />
<img width="1012" height="393" alt="image" src="https://github.com/user-attachments/assets/1cdc5191-c6ae-44e7-bec1-2b4839e74c54" />
<img width="1012" height="393" alt="image" src="https://github.com/user-attachments/assets/0fb69825-9a96-49d2-a07e-62b3112ab1c9" />
<img width="958" height="470" alt="image" src="https://github.com/user-attachments/assets/257c84d2-6692-40cb-8e0e-8c81e37b1be8" />
<img width="958" height="470" alt="image" src="https://github.com/user-attachments/assets/c5f1d136-f814-4118-85dd-c4914829635e" />
<img width="958" height="470" alt="image" src="https://github.com/user-attachments/assets/8e2660bb-c9e6-4a64-bcda-f058af545166" />


# RESULT: 
Analysis of audio signal for noise removal is successfully executed in co lab
