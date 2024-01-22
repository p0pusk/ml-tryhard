import pandas as pd
import librosa


def feature_extractor(file):
    num_segment = 1
    num_mfcc = 20
    n_fft = 2048
    hop_length = 512

    features = []
    audio, sample_rate = librosa.load(file.name, res_type="kaiser_fast")
    stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    features.append(stft.mean())
    features.append(stft.var())

    rms = librosa.feature.rms(y=audio)
    features.append(rms.mean())
    features.append(rms.var())

    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    features.append(spectral_centroid.mean())
    features.append(spectral_centroid.var())

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    features.append(spectral_bandwidth.mean())
    features.append(spectral_bandwidth.var())

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    features.append(rolloff.mean())
    features.append(rolloff.var())

    zero_crossings = librosa.feature.zero_crossing_rate(y=audio)
    features.append(zero_crossings.mean())
    features.append(zero_crossings.var())

    harmony, perceptr = librosa.effects.hpss(y=audio)
    features.append(harmony.mean())
    features.append(harmony.var())
    features.append(perceptr.mean())
    features.append(perceptr.var())

    tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
    features.append(tempo)

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sample_rate, n_mfcc=num_mfcc, hop_length=hop_length
    )
    mfcc = mfcc.T

    for x in range(20):
        features.append(mfcc[:, x].mean())
        features.append(mfcc[:, x].var())

    return pd.DataFrame(features).transpose()
