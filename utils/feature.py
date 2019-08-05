import librosa
import wave
import numpy as np
import torch

def load_audio(wav_path, normalize=False):  # -> numpy array
    with wave.open(wav_path) as wav:
        wav = np.frombuffer(wav.readframes(wav.getnframes()), dtype="int16")
        wav = wav.astype("float")
    if normalize:
        wav = (wav - wav.mean()) / wav.std()
    return wav


def spectrogram(wav, 
                sample_rate = 16000,
                window_size = 0.02,
                window_stride = 0.01,
                normalize=True):
    n_fft = int(sample_rate * window_size)
    win_length = n_fft
    hop_length = int(sample_rate * window_stride)
    window = "hamming"

    D = librosa.stft(
        wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window
    )

    spec, phase = librosa.magphase(D)
    spec = np.log1p(spec)
    spec = torch.FloatTensor(spec)

    if normalize:
        spec = (spec - spec.mean()) / spec.std()

    return spec
