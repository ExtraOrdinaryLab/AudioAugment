import random
from typing import Sequence, Union, List, Callable

import torch
import librosa
import torchaudio
import numpy as np
import torch.nn as nn
import torchaudio.compliance.kaldi as ta_kaldi
from augment import EffectChain as WavAugmentEffectChain

from audio_augment.utils import to_numpy, to_tensor

__all__ = [
    'RandomCrop', 'AddWhiteNoise', 'TimeStretch', 'PitchShift', 'PolarityInversion', 
    'RandomGain', 'BitCrush', 'ClippingDistortion', 'Reverb', 'FBank', 
    'SpecAugment', 'Normalize'
]


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


class RandomCrop(object):

    supports_multichannel = True

    def __init__(self, crop_length: float, sample_rate: int = 16000):
        assert crop_length > 0.0, ValueError("`max_length` must be greater than zero.")
        self.crop_length = crop_length
        self.sample_rate = sample_rate

    def __call__(self, audio_data: np.ndarray) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data)
        # The audio clip is shorter than the crop length
        if audio_data.shape[-1] <= int(self.crop_length * self.sample_rate):
            return audio_data
        offset = random.randint(0, audio_data.shape[-1] - self.crop_length * self.sample_rate)
        return audio_data[:, offset:offset + int(self.crop_length * self.sample_rate)]


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


class BitCrush(object):

    supports_multichannel = True

    def __init__(self, min_bit_depth: int = 5, max_bit_depth: int = 10):
        self.min_bit_depth = min_bit_depth
        self.max_bit_depth = max_bit_depth
        assert min_bit_depth >= 1, ValueError("`min_bit_depth` must be at least 1.")
        assert max_bit_depth <= 32, ValueError("`max_bit_depth` must not be greater than 32.")
        assert min_bit_depth < max_bit_depth, ValueError("`min_bit_depth` must be smaller than `max_bit_depth`")

    def __call__(self, audio_data: np.ndarray) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data)
        bit_depth = random.randint(self.min_bit_depth, self.max_bit_depth)
        q = (2 ** bit_depth / 2) + 1
        aug_signal = np.round(audio_data * q) / q
        return aug_signal


class ClippingDistortion(object):

    supports_multichannel = True

    def __init__(
        self,
        min_percentile_threshold: int = 0,
        max_percentile_threshold: int = 40,
    ):
        self.min_percentile_threshold = min_percentile_threshold
        self.max_percentile_threshold = max_percentile_threshold
        assert min_percentile_threshold <= max_percentile_threshold
        assert 0 <= min_percentile_threshold <= 100, ValueError('`min_percentile_threshold` must be smaller than 100.')
        assert 0 <= max_percentile_threshold <= 100, ValueError('`max_percentile_threshold` must be smaller than 100.')

    def __call__(self, audio_data: np.ndarray) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data)
        percentile_threshold = random.randint(
            self.min_percentile_threshold, self.max_percentile_threshold
        )
        lower_percentile_threshold = int(percentile_threshold / 2)
        lower_threshold, upper_threshold = np.percentile(
            audio_data, [lower_percentile_threshold, 100 - lower_percentile_threshold]
        )
        aug_signal = np.clip(audio_data, lower_threshold, upper_threshold)
        return aug_signal


class Reverb(object):

    supports_multichannel = True

    def __init__(
        self, 
        sample_rate: int = 16000, 
        reverberance_min: int = 0, 
        reverberance_max: int = 100, 
        dumping_factor_min: int = 0, 
        dumping_factor_max: int = 100, 
        room_size_min: int = 0, 
        room_size_max: int = 100, 
    ):
        self.sample_rate = sample_rate
        self.reverberance_min = reverberance_min
        self.reverberance_max = reverberance_max
        self.dumping_factor_min = dumping_factor_min
        self.dumping_factor_max = dumping_factor_max
        self.room_size_min = room_size_min
        self.room_size_max = room_size_max

        self.source_info = {'rate': self.sample_rate}
        self.target_info = {'channel': 1, 'rate': self.sample_rate}

    def __call__(self, audio_data: np.ndarray) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data)
        reverberance = random.randint(self.reverberance_min, self.reverberance_max)
        dumping_factor = random.randint(self.dumping_factor_min, self.dumping_factor_max)
        room_size = random.randint(self.room_size_min, self.room_size_max)
        num_channels = audio_data.shape[0]
        effect_chain = (
            WavAugmentEffectChain()
            .reverb(reverberance, dumping_factor, room_size)
            .channels(num_channels)
        )
        aug_signal = effect_chain.apply(
            to_tensor(audio_data, device='cpu').to(torch.float32), 
            src_info={'rate': self.sample_rate}, 
            target_info={'channel': num_channels, 'rate': self.sample_rate}
        )
        aug_signal = to_numpy(aug_signal)
        return aug_signal


class FBank(object):

    supports_multichannel = False

    def __init__(
        self, 
        sampling_rate: int = 16000, 
        num_mel_bins: int = 128, 
        max_frame_length: int = 1024, 
        frame_length: float = 25,
        frame_shift: float = 10,
    ):
        self.sampling_rate = sampling_rate
        self.num_mel_bins = num_mel_bins
        self.max_frame_length = max_frame_length
        self.frame_length = frame_length
        self.frame_shift = frame_shift

    @staticmethod
    def _extract_fbank(
        waveform: np.ndarray, 
        sampling_rate: int = 16000, 
        max_frame_length: int = 1024, 
        num_mel_bins: int = 128, 
        frame_length: float = 25,
        frame_shift: float = 10,
    ) -> np.ndarray:
        waveform = to_tensor(waveform, device='cpu').float()
        fbank = ta_kaldi.fbank(
            waveform,
            sample_frequency=sampling_rate,
            window_type="hanning",
            num_mel_bins=num_mel_bins,
            frame_length=frame_length, 
            frame_shift=frame_shift, 
        )
        num_frames = fbank.shape[0]
        difference = max_frame_length - num_frames
        
        if difference > 0:
            pad_module = nn.ZeroPad2d((0, 0, 0, difference))
            fbank = pad_module(fbank)
        elif difference < 0:
            fbank = fbank[0:max_frame_length, :]

        fbank = fbank.numpy() # (num_frames, num_mel_bins)
        return fbank

    def __call__(self, audio_data: np.ndarray) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data)
        fbank = self._extract_fbank(
            audio_data, 
            sampling_rate=self.sampling_rate, 
            max_frame_length=self.max_frame_length, 
            num_mel_bins=self.num_mel_bins, 
            frame_length=self.frame_length, 
            frame_shift=self.frame_shift
        )
        return fbank
    

class SpecAugment(object):

    supports_multichannel = False

    def __init__(
        self, 
        freq_mask_param: int = 48, 
        time_mask_param: int = 192, 
    ):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

    def __call__(self, fbank: np.ndarray) -> np.ndarray:
        """
        fbank : np.ndarray
            Spectrogram of shape (num_frames, num_mel_bins)
        """
        fbank = to_tensor(fbank, device='cpu')
        if self.freq_mask_param != 0:
            freqm = torchaudio.transforms.FrequencyMasking(self.freq_mask_param)
            fbank = freqm(fbank.transpose(0, 1).unsqueeze(0))
            fbank = fbank.squeeze(0).transpose(0, 1)
        if self.time_mask_param != 0:
            timem = torchaudio.transforms.TimeMasking(self.time_mask_param)
            fbank = timem(fbank.transpose(0, 1).unsqueeze(0))
            fbank = fbank.squeeze(0).transpose(0, 1)
        fbank = to_numpy(fbank)
        return fbank


class Normalize(object):

    def __init__(
        self, 
        mean: float = -4.2677393, 
        std: float = 4.5689974, 
    ):
        self.mean = mean
        self.std = std

    def __call__(self, input_values: np.ndarray) -> np.ndarray:
        return (input_values - (self.mean)) / (self.std * 2)


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
    
    def __call__(self, audio_data: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            audio_data = t(audio_data)
        return audio_data


class MultiViewTransform(object):
    """Transforms an image into multiple views.

    Args:
        transforms:
            A sequence of transforms. Every transform creates a new view.

    """
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = transforms

    def __call__(self, input_values: Union[torch.Tensor, np.ndarray]) -> Union[List[torch.Tensor], List[np.ndarray]]:
        return [transform(input_values) for transform in self.transforms]


class ToOneHot(object):

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def _one_hot_transform(self, labels: List[int], num_classes: int) -> np.ndarray:
        one_hot_tensor = np.zeros((num_classes, ), dtype=float)
        one_hot_tensor[labels] = 1
        return one_hot_tensor

    def __call__(self, labels):
        return self._one_hot_transform(labels, self.num_classes)