import random
from abc import ABC, abstractmethod
from typing import Sequence, Union, List, Callable

import librosa
import numpy_rms
import scipy.signal
import numpy as np
import soundfile as sf

import torch
import torchaudio
import torch.nn as nn
import torchaudio.compliance.kaldi as ta_kaldi
from augment import EffectChain as WavAugmentEffectChain

from audio_augment.utils import to_numpy, to_tensor


def reshape_audio_clip(waveform: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Reshape the input ndarray representing an audio clip waveform to [num_channels, num_samples].
    
    Parameters:
        waveform (ndarray): Input audio clip waveform.
        
    Returns:
        ndarray: Reshaped audio clip waveform.
    """
    waveform = to_numpy(waveform)

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


class DefaultReprMixin:
    """
    Mixin that provides an automatic __repr__ using the instance's __dict__.
    """
    def __repr__(self) -> str:
        classname = self.__class__.__name__
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{classname}({params})"


class AudioTransform(ABC, DefaultReprMixin):
    """
    Abstract base class for audio transforms that should have a uniform interface.
    Classes inheriting from AudioTransform must implement __call__.
    """
    supports_multichannel: bool = True

    @abstractmethod
    def __call__(self, audio_data: np.ndarray) -> np.ndarray:
        pass


class RandomCrop(AudioTransform):
    """
    Randomly crop an audio clip to a specified length.
    """
    supports_multichannel = True

    def __init__(self, crop_length: float, sample_rate: int = 16000):
        """
        Initialize RandomCrop.

        Parameters:
            crop_length (float): Duration of the crop in seconds.
            sample_rate (int): Sampling rate of the audio.
        """
        assert crop_length > 0.0, ValueError("`max_length` must be greater than zero.")
        self.crop_length = crop_length
        self.sample_rate = sample_rate

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data)
        # The audio clip is shorter than the crop length
        if audio_data.shape[-1] <= int(self.crop_length * self.sample_rate):
            return audio_data
        offset = random.randint(0, audio_data.shape[-1] - self.crop_length * self.sample_rate)
        return audio_data[:, offset:offset + int(self.crop_length * self.sample_rate)]


class AddWhiteNoise(AudioTransform):
    """
    Add white noise to an audio clip.
    """
    supports_multichannel = True

    def __init__(self, min_amplitude: float = 0.001, max_amplitude: float = 0.015):
        """
        Initialize AddWhiteNoise.

        Parameters:
            min_amplitude (float): Minimum amplitude of the noise.
            max_amplitude (float): Maximum amplitude of the noise.
        """
        assert min_amplitude > 0.0, ValueError("`min_amplitude` must be greater than zero.")
        assert max_amplitude > 0.0, ValueError("`max_amplitude` must be greater than zero.")
        assert max_amplitude >= min_amplitude, ValueError("`max_amplitude` must be greater than `min_amplitude`.")
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data)
        amplitude = random.uniform(self.min_amplitude, self.max_amplitude)
        noise = np.random.normal(0, audio_data.std(), size=audio_data.shape)
        aug_signal = audio_data + noise * amplitude
        return aug_signal


class AddNoiseFromFiles(AudioTransform):
    """
    Add noise from external files to an audio clip at a random Signal-to-Noise Ratio (SNR).
    """
    supports_multichannel = False

    def __init__(self, noise_files, snr_low: int = 0, snr_high: int = 10, sample_rate: int = 16000, normalize: bool = False):
        """
        Initialize AddNoiseFromFiles.

        Parameters:
            noise_files (list): List of paths to noise audio files.
            snr_low (int): Lower bound for SNR in dB.
            snr_high (int): Upper bound for SNR in dB.
            sample_rate (int): Sampling rate of the audio.
            normalize (bool): Whether to normalize the output audio.
        """
        self.noise_files = noise_files
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.sample_rate = sample_rate
        self.normalize = normalize

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data) # (1, T)

        noise_path = random.choice(self.noise_files)
        noise, noise_sr = sf.read(noise_path, dtype='float32')

        if noise_sr!= self.sample_rate:
            noise = librosa.resample(noise, orig_sr=noise_sr, target_sr=self.sample_rate)

        # Ensure noise is at least as long as the waveform
        T = audio_data.shape[1]
        if len(noise) < T:
            repeat_factor = int(np.ceil(T / len(noise)))
            noise = np.tile(noise, repeat_factor)[:T]
        else:
            start_idx = random.randint(0, len(noise) - T)
            noise = noise[start_idx:start_idx + T]

        # Compute signal and noise power
        signal_power = np.mean(audio_data ** 2)
        noise_power = np.mean(noise ** 2)

        # Randomly select an SNR level
        snr_db = random.uniform(self.snr_low, self.snr_high)
        snr = 10 ** (snr_db / 10)
        noise = noise * np.sqrt(signal_power / (noise_power * snr))

        # Add noise to signal
        noisy_waveform = audio_data + reshape_audio_clip(noise)

        # Normalize if required
        if self.normalize:
            max_val = np.max(np.abs(noisy_waveform))
            if max_val > 1:
                noisy_waveform /= max_val

        return noisy_waveform

    def __repr__(self) -> str:
        classname = self.__class__.__name__
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items() if k != 'noise_files')
        return f"{classname}({params})"


class AddReverbFromFiles(AudioTransform):
    """
    Convolve an audio signal with an impulse response from a file to add reverb.

    Parameters
    ----------
    rir_files : list
        List of paths to impulse response audio files.
    rir_scale_factor : float
        Scaling factor for the impulse response duration. If 0 < scale_factor < 1, the impulse response
        is compressed (less reverb), while if scale_factor > 1 it is dilated (more reverb).
    """
    supports_multichannel = False

    def __init__(self, rir_files, sample_rate: int = 16000, rir_scale_factor: float = 1.0):
        """
        Initialize AddReverbFromFiles.

        Parameters:
            rir_files (list): List of paths to impulse response files.
            sample_rate (int): Sampling rate of the audio.
            rir_scale_factor (float): Scale factor for the impulse response.
        """
        self.rir_files = rir_files
        self.sample_rate = sample_rate
        self.rir_scale_factor = rir_scale_factor

    def __repr__(self) -> str:
        classname = self.__class__.__name__
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items() if k != 'rir_files')
        return f"{classname}({params})"

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data)

        # Load a random impulse response file
        rir_path = random.choice(self.rir_files)
        rir, rir_sr = sf.read(rir_path, dtype='float32')

        # Convert to mono if necessary
        if len(rir.shape) > 1:
            rir = np.mean(rir, axis=1)

        # Resample if necessary
        if rir_sr != self.sample_rate:
            rir = librosa.resample(rir, orig_sr=rir_sr, target_sr=self.sample_rate)

        # Scale the RIR if needed
        if self.rir_scale_factor != 1:
            num_samples = int(len(rir) * self.rir_scale_factor)
            rir = scipy.signal.resample(rir, num_samples)

        # Normalize RIR energy
        rir = rir / np.max(np.abs(rir))
        
        # Convolve the input waveform with the RIR
        reverberated_waveform = scipy.signal.fftconvolve(audio_data, rir[None, :], mode='full')
        
        # Truncate to match original length
        reverberated_waveform = reverberated_waveform[:, :audio_data.shape[1]]

        return reverberated_waveform


class TimeStretch(AudioTransform):
    """
    Time-stretching transform.

    Stretch factor:
    - If rate > 1, then the signal is sped up (shorter).
    - If rate < 1, then the signal is slowed down (longer).
    """
    supports_multichannel = True

    def __init__(self, min_rate: float = 0.8, max_rate: float = 1.25, leave_length_unchanged: bool = True):
        """
        Initialize TimeStretch.

        Parameters:
            min_rate (float): Minimum stretch rate.
            max_rate (float): Maximum stretch rate.
            leave_length_unchanged (bool): If True, the output length is adjusted to match the original.
        """
        assert min_rate >= 0.1, ValueError("`min_rate` must be greater than 0.1")
        assert max_rate <= 10, ValueError("`max_rate` must be smaller than 10")
        assert min_rate <= max_rate, ValueError("`min_rate` must be smaller than `max_rate`")
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.leave_length_unchanged = leave_length_unchanged

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        # Ensure the audio is in the correct shape (this function should handle reshaping)
        audio_data = reshape_audio_clip(audio_data)
        stretch_factor = random.uniform(self.min_rate, self.max_rate)
        aug_signal = librosa.effects.time_stretch(audio_data, rate=stretch_factor)

        if self.leave_length_unchanged:
            target_length = audio_data.shape[-1]
            current_length = aug_signal.shape[-1]

            if current_length < target_length:
                # If the stretched signal is shorter, pad with zeros
                padded_samples = np.zeros_like(audio_data)
                padded_samples[..., :current_length] = aug_signal
                aug_signal = padded_samples
            elif current_length > target_length:
                # If the stretched signal is longer, truncate to the target length
                aug_signal = aug_signal[..., :target_length]

        return aug_signal


class PitchShift(AudioTransform):
    """
    Pitch shift the sound up or down without changing the tempo.
    """
    supports_multichannel = True

    def __init__(self, sample_rate: int = 16000, min_semitones: float = -4.0, max_semitones: float = 4.0):
        """
        Initialize PitchShift.

        Parameters:
            sample_rate (int): Sampling rate of the audio.
            min_semitones (float): Minimum semitones to shift.
            max_semitones (float): Maximum semitones to shift.
        """
        assert min_semitones >= -12, ValueError("`min_semitones` must be greater than -12")
        assert max_semitones <= 12, ValueError("`max_semitones` must be smaller than 12")
        assert min_semitones <= max_semitones, ValueError("`max_semitones` must be greater than `min_semitones`")
        self.sample_rate = sample_rate
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data)
        num_semitones = random.uniform(self.min_semitones, self.max_semitones)
        aug_signal = librosa.effects.pitch_shift(
            audio_data, sr=self.sample_rate, n_steps=num_semitones
        )
        return aug_signal


class PolarityInversion(AudioTransform):
    """Flip the audio samples upside-down, reversing their polarity"""

    supports_multichannel = True

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data)
        aug_signal = np.negative(audio_data)
        return aug_signal


class RandomGain(AudioTransform):
    """
    Multiply the audio by a random amplitude factor to adjust the volume.
    """
    supports_multichannel = True

    def __init__(self, min_gain_db: float = None, max_gain_db: float = None):
        """
        Initialize RandomGain.

        Parameters:
            min_gain_db (float): Minimum gain in decibels.
            max_gain_db (float): Maximum gain in decibels.
        """
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

        if self.min_gain_db is None:
            self.min_gain_db = -12.0
        if self.max_gain_db is None:
            self.max_gain_db = 12.0

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data)
        gain_db = np.random.uniform(self.min_gain_db, self.max_gain_db)
        amplitude_ratio = self.convert_decibels_to_amplitude_ratio(gain_db)
        aug_signal = audio_data * amplitude_ratio
        return aug_signal

    @staticmethod
    def convert_decibels_to_amplitude_ratio(decibels):
        return 10 ** (decibels / 20)


class BitCrush(AudioTransform):
    """
    Reduce the bit depth of the audio, creating a bit-crushed effect.
    """
    supports_multichannel = True

    def __init__(self, min_bit_depth: int = 5, max_bit_depth: int = 10):
        """
        Initialize BitCrush.

        Parameters:
            min_bit_depth (int): Minimum bit depth.
            max_bit_depth (int): Maximum bit depth.
        """
        self.min_bit_depth = min_bit_depth
        self.max_bit_depth = max_bit_depth
        assert min_bit_depth >= 1, ValueError("`min_bit_depth` must be at least 1.")
        assert max_bit_depth <= 32, ValueError("`max_bit_depth` must not be greater than 32.")
        assert min_bit_depth < max_bit_depth, ValueError("`min_bit_depth` must be smaller than `max_bit_depth`")

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data)
        bit_depth = random.randint(self.min_bit_depth, self.max_bit_depth)
        q = (2 ** bit_depth / 2) + 1
        aug_signal = np.round(audio_data * q) / q
        return aug_signal


class ClippingDistortion(AudioTransform):
    """
    Apply clipping distortion by clipping the audio signal based on a percentile threshold.
    """
    supports_multichannel = True

    def __init__(
        self,
        min_percentile_threshold: int = 0,
        max_percentile_threshold: int = 40,
    ):
        """
        Initialize ClippingDistortion.

        Parameters:
            min_percentile_threshold (int): Minimum percentile threshold.
            max_percentile_threshold (int): Maximum percentile threshold.
        """
        self.min_percentile_threshold = min_percentile_threshold
        self.max_percentile_threshold = max_percentile_threshold
        assert min_percentile_threshold <= max_percentile_threshold
        assert 0 <= min_percentile_threshold <= 100, ValueError('`min_percentile_threshold` must be smaller than 100.')
        assert 0 <= max_percentile_threshold <= 100, ValueError('`max_percentile_threshold` must be smaller than 100.')

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
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


class Reverb(AudioTransform):
    """
    Apply reverberation effect to audio data using WavAugment.
    """
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
        """
        Initialize Reverb.

        Parameters:
            sample_rate (int): Sampling rate of the audio.
            reverberance_min (int): Minimum reverberance percentage.
            reverberance_max (int): Maximum reverberance percentage.
            dumping_factor_min (int): Minimum dumping factor percentage.
            dumping_factor_max (int): Maximum dumping factor percentage.
            room_size_min (int): Minimum room size percentage.
            room_size_max (int): Maximum room size percentage.
        """
        self.sample_rate = sample_rate
        self.reverberance_min = reverberance_min
        self.reverberance_max = reverberance_max
        self.dumping_factor_min = dumping_factor_min
        self.dumping_factor_max = dumping_factor_max
        self.room_size_min = room_size_min
        self.room_size_max = room_size_max

        self.source_info = {'rate': self.sample_rate}
        self.target_info = {'channel': 1, 'rate': self.sample_rate}

    def __repr__(self) -> str:
        classname = self.__class__.__name__
        info = ['source_info', 'target_info']
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items() if k not in info)
        return f"{classname}({params})"

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
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


class TanhDistortion(AudioTransform):
    """
    Apply hyperbolic tangent distortion to the audio signal.
    """
    supports_multichannel = True

    def __init__(
        self, min_distortion: float = 0.01, max_distortion: float = 0.7
    ):
        """
        Initialize TanhDistortion.

        Parameters:
            min_distortion (float): Minimum distortion amount.
            max_distortion (float): Maximum distortion amount.
        """
        assert 0 <= min_distortion <= 1
        assert 0 <= max_distortion <= 1
        assert min_distortion <= max_distortion
        self.min_distortion = min_distortion
        self.max_distortion = max_distortion

    @staticmethod
    def calculate_rms(samples):
        """Given a numpy array of audio samples, return its Root Mean Square (RMS)."""
        return np.mean(numpy_rms.rms(samples))

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data)
        distortion_amount = random.uniform(
            self.min_distortion, self.max_distortion
        )
        percentile = 100 - 99 * distortion_amount
        threshold = np.percentile(np.abs(audio_data), percentile)
        gain_factor = 0.5 / (threshold + 1e-6)
        aug_signal = np.tanh(gain_factor * audio_data)
        # Scale the output so its loudness matches the input
        rms_before = self.calculate_rms(audio_data)
        if rms_before > 1e-9:
            rms_after = self.calculate_rms(aug_signal)
            post_gain = rms_before / rms_after
            aug_signal = post_gain * aug_signal
        return aug_signal


class EchoEffect(AudioTransform):
    """
    Apply an echo effect by mixing a delayed version of the audio with the original.
    """
    supports_multichannel = True

    def __init__(self, delay: float = 0.3, decay: float = 0.5, sample_rate: int = 16000):
        """
        Initialize EchoEffect.

        Parameters:
            delay (float): Delay in seconds before the echo starts.
            decay (float): Decay factor applied to the echo.
            sample_rate (int): Sampling rate of the audio.
        """
        self.delay = delay
        self.decay = decay
        self.sample_rate = sample_rate

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Apply the echo effect to the audio data.

        Parameters:
            audio_data (ndarray or Tensor): Input audio data.
        
        Returns:
            ndarray: Audio data with echo effect applied.
        """
        audio_data = reshape_audio_clip(audio_data)
        delay_samples = int(self.delay * self.sample_rate)
        echo = np.zeros_like(audio_data)
        if audio_data.shape[1] > delay_samples:
            echo[:, delay_samples:] = audio_data[:, :-delay_samples] * self.decay
        return audio_data + echo


class LowPassFilter(AudioTransform):
    """
    Apply a low-pass filter to remove high-frequency content from the audio signal.
    """
    supports_multichannel = True

    def __init__(self, cutoff_freq: float = 4000, sample_rate: int = 16000, order: int = 5):
        """
        Initialize LowPassFilter.

        Parameters:
            cutoff_freq (float): Cutoff frequency in Hz.
            sample_rate (int): Sampling rate of the audio.
            order (int): Order of the Butterworth filter.
        """
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.order = order

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Apply low-pass filtering to the audio data.

        Parameters:
            audio_data (ndarray or Tensor): Input audio data.
        
        Returns:
            ndarray: Low-pass filtered audio data.
        """
        audio_data = reshape_audio_clip(audio_data)
        nyq = 0.5 * self.sample_rate
        normal_cutoff = self.cutoff_freq / nyq
        b, a = scipy.signal.butter(self.order, normal_cutoff, btype='low', analog=False)
        filtered = scipy.signal.filtfilt(b, a, audio_data, axis=1)
        return filtered


class HighPassFilter(AudioTransform):
    """
    Apply a high-pass filter to remove low-frequency content from the audio signal.
    """
    supports_multichannel = True

    def __init__(self, cutoff_freq: float = 300, sample_rate: int = 16000, order: int = 5):
        """
        Initialize HighPassFilter.

        Parameters:
            cutoff_freq (float): Cutoff frequency in Hz.
            sample_rate (int): Sampling rate of the audio.
            order (int): Order of the Butterworth filter.
        """
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.order = order

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Apply high-pass filtering to the audio data.

        Parameters:
            audio_data (ndarray or Tensor): Input audio data.
        
        Returns:
            ndarray: High-pass filtered audio data.
        """
        audio_data = reshape_audio_clip(audio_data)
        nyq = 0.5 * self.sample_rate
        normal_cutoff = self.cutoff_freq / nyq
        b, a = scipy.signal.butter(self.order, normal_cutoff, btype='high', analog=False)
        filtered = scipy.signal.filtfilt(b, a, audio_data, axis=1)
        return filtered


class BandPassFilter(AudioTransform):
    """
    Apply a band-pass filter to isolate a specific frequency band from the audio signal.
    """
    supports_multichannel = True

    def __init__(self, low_cutoff: float = 300, high_cutoff: float = 3400, sample_rate: int = 16000, order: int = 4):
        """
        Initialize BandPassFilter.

        Parameters:
            low_cutoff (float): Lower cutoff frequency in Hz.
            high_cutoff (float): Upper cutoff frequency in Hz.
            sample_rate (int): Sampling rate of the audio.
            order (int): Order of the Butterworth filter.
        """
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.sample_rate = sample_rate
        self.order = order

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Apply band-pass filtering to the audio data.

        Parameters:
            audio_data (ndarray or Tensor): Input audio data.
        
        Returns:
            ndarray: Band-pass filtered audio data.
        """
        audio_data = reshape_audio_clip(audio_data)
        nyq = 0.5 * self.sample_rate
        low = self.low_cutoff / nyq
        high = self.high_cutoff / nyq
        b, a = scipy.signal.butter(self.order, [low, high], btype='band')
        filtered = scipy.signal.filtfilt(b, a, audio_data, axis=1)
        return filtered


class ReverseAudio(AudioTransform):
    """
    Reverse the audio signal in time.
    """
    supports_multichannel = True

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Reverse the audio data.

        Parameters:
            audio_data (ndarray or Tensor): Input audio data.
        
        Returns:
            ndarray: Time-reversed audio data.
        """
        audio_data = reshape_audio_clip(audio_data)
        return audio_data[:, ::-1]


class DynamicRangeCompression(AudioTransform):
    """
    Apply dynamic range compression to reduce the difference between loud and soft parts of the audio.
    """
    supports_multichannel = True

    def __init__(self, threshold: float = 0.5, ratio: float = 4.0):
        """
        Initialize DynamicRangeCompression.

        Parameters:
            threshold (float): Amplitude threshold above which compression is applied.
            ratio (float): Compression ratio.
        """
        self.threshold = threshold
        self.ratio = ratio

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Compress the dynamic range of the audio data.

        Parameters:
            audio_data (ndarray or Tensor): Input audio data.
        
        Returns:
            ndarray: Audio data after dynamic range compression.
        """
        audio_data = reshape_audio_clip(audio_data)
        abs_audio = np.abs(audio_data)
        mask = abs_audio > self.threshold
        compressed = np.copy(audio_data)
        compressed[mask] = np.sign(audio_data[mask]) * (
            self.threshold + (abs_audio[mask] - self.threshold) / self.ratio
        )
        return compressed


class Equalizer(AudioTransform):
    """
    Apply a simple equalizer by boosting or attenuating specified frequency bands.
    """
    supports_multichannel = True

    def __init__(self, gains: dict, sample_rate: int = 16000):
        """
        Initialize Equalizer.

        Parameters:
            gains (dict): Dictionary with center frequencies (Hz) as keys and gain factors as values.
            sample_rate (int): Sampling rate of the audio.
        """
        self.gains = gains
        self.sample_rate = sample_rate

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Apply equalization to the audio data.

        Parameters:
            audio_data (ndarray or Tensor): Input audio data.
        
        Returns:
            ndarray: Equalized audio data.
        """
        audio_data = reshape_audio_clip(audio_data)
        equalized = np.zeros_like(audio_data)
        for freq, gain in self.gains.items():
            low = max(freq - 50, 0)
            high = freq + 50
            nyq = 0.5 * self.sample_rate
            low_norm = low / nyq
            high_norm = high / nyq
            b, a = scipy.signal.butter(2, [low_norm, high_norm], btype='band')
            filtered = scipy.signal.filtfilt(b, a, audio_data, axis=1)
            equalized += gain * filtered
        return audio_data + equalized


class Flanger(AudioTransform):
    """
    Apply a flanger effect by mixing a time-varying delayed copy of the audio with the original.
    """
    supports_multichannel = True

    def __init__(self, max_delay: float = 0.005, depth: float = 0.002, rate: float = 0.25, sample_rate: int = 16000):
        """
        Initialize Flanger.

        Parameters:
            max_delay (float): Maximum delay in seconds.
            depth (float): Depth of modulation.
            rate (float): Modulation rate in Hz.
            sample_rate (int): Sampling rate of the audio.
        """
        self.max_delay = max_delay
        self.depth = depth
        self.rate = rate
        self.sample_rate = sample_rate

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Apply the flanger effect to the audio data.

        Parameters:
            audio_data (ndarray or Tensor): Input audio data.
        
        Returns:
            ndarray: Audio data with the flanger effect applied.
        """
        audio_data = reshape_audio_clip(audio_data)
        num_samples = audio_data.shape[1]
        t = np.arange(num_samples) / self.sample_rate
        mod = self.depth * np.sin(2 * np.pi * self.rate * t)
        delay_samples = (self.max_delay * self.sample_rate * (1 + mod)).astype(int)
        flanged = np.copy(audio_data)
        for n in range(num_samples):
            d = delay_samples[n]
            if n - d >= 0:
                flanged[:, n] = audio_data[:, n] + audio_data[:, n - d]
        return flanged


class Chorus(AudioTransform):
    """
    Apply a chorus effect by mixing several delayed copies of the audio signal.
    """
    supports_multichannel = True

    def __init__(self, delays: list = [0.02, 0.025, 0.03], decays: list = [0.5, 0.5, 0.5], sample_rate: int = 16000):
        """
        Initialize Chorus.

        Parameters:
            delays (list): List of delay times in seconds.
            decays (list): List of decay factors corresponding to each delay.
            sample_rate (int): Sampling rate of the audio.
        """
        self.delays = delays
        self.decays = decays
        self.sample_rate = sample_rate

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Apply the chorus effect to the audio data.

        Parameters:
            audio_data (ndarray or Tensor): Input audio data.
        
        Returns:
            ndarray: Audio data with the chorus effect applied.
        """
        audio_data = reshape_audio_clip(audio_data)
        chorus_signal = np.copy(audio_data)
        for delay, decay in zip(self.delays, self.decays):
            delay_samples = int(delay * self.sample_rate)
            temp = np.zeros_like(audio_data)
            if audio_data.shape[1] > delay_samples:
                temp[:, delay_samples:] = audio_data[:, :-delay_samples] * decay
            chorus_signal += temp
        return chorus_signal


class Tremolo(AudioTransform):
    """
    Apply a tremolo effect by modulating the amplitude of the audio signal.
    """
    supports_multichannel = True

    def __init__(self, rate: float = 5.0, depth: float = 0.5, sample_rate: int = 16000):
        """
        Initialize Tremolo.

        Parameters:
            rate (float): Modulation rate in Hz.
            depth (float): Modulation depth (range 0 to 1).
            sample_rate (int): Sampling rate of the audio.
        """
        self.rate = rate
        self.depth = depth
        self.sample_rate = sample_rate

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Apply the tremolo effect to the audio data.

        Parameters:
            audio_data (ndarray or Tensor): Input audio data.
        
        Returns:
            ndarray: Audio data with tremolo effect applied.
        """
        audio_data = reshape_audio_clip(audio_data)
        num_samples = audio_data.shape[1]
        t = np.linspace(0, num_samples / self.sample_rate, num_samples)
        modulation = (1 + self.depth * np.sin(2 * np.pi * self.rate * t)) / 2
        return audio_data * modulation


class FBank(AudioTransform):

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

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        audio_data = reshape_audio_clip(audio_data)
        fbank = self._extract_fbank(
            audio_data, 
            sampling_rate=self.sampling_rate, 
            max_frame_length=self.max_frame_length, 
            num_mel_bins=self.num_mel_bins, 
            frame_length=self.frame_length, 
            frame_shift=self.frame_shift
        )
        assert fbank.shape == (self.max_frame_length, self.num_mel_bins)
        return fbank
    

class SpecAugment(AudioTransform):

    supports_multichannel = False

    def __init__(
        self, 
        freq_mask_param: int = 48, 
        time_mask_param: int = 192, 
    ):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

    def __call__(self, input_values: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        input_values : np.ndarray
            Spectrogram of shape (num_frames, num_mel_bins)
        """
        assert input_values.ndim == 2
        input_values = to_tensor(input_values, device='cpu')
        if self.freq_mask_param != 0:
            freqm = torchaudio.transforms.FrequencyMasking(self.freq_mask_param)
            input_values = freqm(input_values.transpose(0, 1).unsqueeze(0))
            input_values = input_values.squeeze(0).transpose(0, 1)
        if self.time_mask_param != 0:
            timem = torchaudio.transforms.TimeMasking(self.time_mask_param)
            input_values = timem(input_values.transpose(0, 1).unsqueeze(0))
            input_values = input_values.squeeze(0).transpose(0, 1)
        input_values = to_numpy(input_values)
        return input_values


class CutOut(AudioTransform):

    def __init__(self, num_holes, length):
        self.num_holes = num_holes
        self.length = length

    def __call__(self, input_values: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        input_values : np.ndarray
            Tensor input shape (num_frames, num_mel_bins)
        """
        assert input_values.ndim == 2
        h, w = input_values.shape
        mask = np.ones((h, w), np.float32)

        for n in range(self.num_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1: y2, x1: x2] = 0.0
        
        input_values = input_values * mask

        return input_values


class Normalize(AudioTransform):

    def __init__(
        self, 
        mean: float = -4.2677393, 
        std: float = 4.5689974, 
    ):
        self.mean = mean
        self.std = std

    def __call__(self, input_values: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        input_values = reshape_audio_clip(input_values)
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
    def __init__(self, transform, p: float = 1.0):
        self.transform = transform
        self.p = p

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if self.p < random.random():
            return audio_data
        audio_data = self.transform(audio_data)
        return audio_data


class RandomChoiceTransform(object):
    """
    Applies a list of transformations randomly with a given probability.

    Parameters
    ----------
    transforms : list
        A list of transformation objects to be applied randomly.
    p : float
        The probability of applying any given transformation.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        transform = random.choice(self.transforms)
        audio_data = transform(audio_data)
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

    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for idx in order:
            audio_data = self.transforms[idx](audio_data)
        return audio_data


class Compose(object):

    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
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
