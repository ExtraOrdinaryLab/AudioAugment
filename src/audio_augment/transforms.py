import random

import librosa
import numpy as np


def reshape_audio_clip(waveform: np.ndarray) -> np.ndarray:
    """
    Reshape the input ndarray representing an audio clip waveform to [num_channels, num_samples].
    
    Parameters:
        waveform (ndarray): Input audio clip waveform.
        
    Returns:
        ndarray: Reshaped audio clip waveform.
    """
    if not isinstance(waveform, np.ndarray):
        raise ValueError("`waveform` must be a numpy array.")

    if waveform.ndim == 2:
        # Already in shape (num_channels, num_samples)
        return waveform
    elif waveform.ndim == 1:
        # Mono audio, reshape to (1, num_samples)
        return waveform.reshape((1, -1))
    else:
        raise ValueError("`waveform` must have either 1 or 2 dimensions.")


class AddWhiteNoise(object):

    supports_multichannel = True

    def __init__(self, min_amplitude: float = 0.001, max_amplitude: float = 0.015):
        assert min_amplitude > 0.0, ValueError("`min_amplitude` must be greater than zero.")
        assert max_amplitude > 0.0, ValueError("`max_amplitude` must be greater than zero.")
        assert max_amplitude >= min_amplitude, ValueError("`max_amplitude` must be greater than `min_amplitude`.")
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def __call__(self, audio_data: np.ndarray) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data)
        amplitude = random.uniform(self.min_amplitude, self.max_amplitude)
        noise = np.random.normal(0, audio_data.std(), size=audio_data.shape)
        aug_signal = audio_data + noise * amplitude
        return aug_signal

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_amplitude={self.min_amplitude}, max_amplitude={self.max_amplitude})"


class TimeStretch(object):
    """
    Stretch factor. If rate > 1, then the signal is sped up. If rate < 1, then the signal is slowed down.
    """

    supports_multichannel = True

    def __init__(self, min_rate: float = 0.8, max_rate: float = 1.25, leave_length_unchanged: bool = True):
        assert min_rate >= 0.1, ValueError("`min_rate` must be greater than 0.1")
        assert max_rate <= 10, ValueError("`max_rate` must be smaller than 10")
        assert min_rate <= max_rate, ValueError("`min_rate` must be smaller than `max_rate`")
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.leave_length_unchanged = leave_length_unchanged

    def __call__(self, audio_data: np.ndarray) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data)
        stretch_factor = random.uniform(self.min_rate, self.max_rate)
        aug_signal = librosa.effects.time_stretch(audio_data, rate=stretch_factor)
        if self.leave_length_unchanged:
            padded_samples = np.zeros(shape=aug_signal.shape, dtype=aug_signal.dtype)
            window = aug_signal[..., :aug_signal.shape[-1]]
            actual_window_length = window.shape[-1]  # may be smaller than samples.shape[-1]
            padded_samples[..., :actual_window_length] = window
            aug_signal = padded_samples
        return aug_signal

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(min_rate={self.min_rate}, "
            f"max_rate={self.max_rate}, "
            f"leave_length_unchanged={self.leave_length_unchanged})"
        )


class PitchShift(object):
    """Pitch shift the sound up or down without changing the tempo"""

    supports_multichannel = True

    def __init__(self, sample_rate: int = 16000, min_semitones: float = -4.0, max_semitones: float = 4.0):
        assert min_semitones >= -12, ValueError("`min_semitones` must be greater than -12")
        assert max_semitones <= 12, ValueError("`max_semitones` must be smaller than 12")
        assert min_semitones <= max_semitones, ValueError("`max_semitones` must be greater than `min_semitones`")
        self.sample_rate = sample_rate
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones

    def __call__(self, audio_data: np.ndarray) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data)
        num_semitones = random.uniform(self.min_semitones, self.max_semitones)
        aug_signal = librosa.effects.pitch_shift(
            audio_data, sr=self.sample_rate, n_steps=num_semitones
        )
        return aug_signal

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sample_rate={self.sample_rate}, "
            f"min_semitones={self.min_semitones}, "
            f"max_semitones={self.max_semitones})"
        )


class PolarityInversion(object):
    """Flip the audio samples upside-down, reversing their polarity"""

    supports_multichannel = True

    def __call__(self, audio_data: np.ndarray) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data)
        aug_signal = np.negative(audio_data)
        return aug_signal

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class RandomGain(object):
    """Multiply the audio by a random amplitude factor to reduce or increase the volume"""

    supports_multichannel = True

    def __init__(self, min_gain_db: float = None, max_gain_db: float = None,):
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

        if self.min_gain_db is None:
            self.min_gain_db = -12.0
        if self.max_gain_db is None:
            self.max_gain_db = 12.0

    def __call__(self, audio_data: np.ndarray) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data)
        gain_db = np.random.uniform(self.min_gain_db, self.max_gain_db)
        amplitude_ratio = self.convert_decibels_to_amplitude_ratio(gain_db)
        aug_signal = audio_data * amplitude_ratio
        return aug_signal

    @staticmethod
    def convert_decibels_to_amplitude_ratio(decibels):
        return 10 ** (decibels / 20)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_gain_db={self.min_gain_db}, max_gain_db={self.max_gain_db})"


class RandomApply(object):
    """
    Applies a list of transformations randomly with a given probability.

    Parameters
    ----------
    transforms : list
        A list of transformation objects to be applied randomly.
    p : float
        The probability of applying any given transformation.
    """
    def __init__(self, transforms, p: float):
        self.transforms = transforms
        self.p = p

    def __call__(self, audio_data: np.ndarray) -> np.ndarray:
        if self.p < random.random():
            return audio_data
        for t in self.transforms:
            audio_data = t(audio_data)
        return audio_data


class RandomOrderCompose(object):
    """
    A class that applies a list of transformations to an input audio data in a random order.

    Parameters
    ----------
    transforms : list
        A list of transformation objects. Each object must be callable and accept an audio data
        as input and return the transformed audio data.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio_data: np.ndarray) -> np.ndarray:
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for idx in order:
            audio_data = self.transforms[idx](audio_data)
        return audio_data


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, audio_data):
        for t in self.transforms:
            audio_data = t(audio_data)
        return audio_data