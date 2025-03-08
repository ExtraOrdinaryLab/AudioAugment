import numpy as np
import torch
import pytest
import scipy.signal
import soundfile as sf

# Import the transforms and helper function.
# Replace `your_module` with the actual module name if needed.
from ..audio_augment.transforms import (
    reshape_audio_clip,
    RandomCrop,
    AddWhiteNoise,
    AddNoiseFromFiles,
    AddReverbFromFiles,
    TimeStretch,
    PitchShift,
    PolarityInversion,
    RandomGain,
    BitCrush,
    ClippingDistortion,
    Reverb,
    TanhDistortion,
    EchoEffect,
    LowPassFilter,
    HighPassFilter,
    BandPassFilter,
    ReverseAudio,
    DynamicRangeCompression,
    Equalizer,
    Flanger,
    Chorus,
    Tremolo, 
    FBank,
    SpecAugment,
    CutOut,
    Normalize,
    RandomApply,
    RandomChoiceTransform,
    RandomOrderCompose,
    Compose,
    MultiViewTransform,
    ToOneHot,
)


def test_reshape_audio_clip(dummy_audio):
    reshaped = reshape_audio_clip(dummy_audio)
    assert reshaped.ndim == 2
    assert reshaped.shape == dummy_audio.shape


# -------------------------------
# Fixtures for dummy inputs
# -------------------------------

@pytest.fixture
def dummy_audio():
    """
    Create a dummy 1-second mono sine wave at 440Hz sampled at 16kHz.
    """
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio.reshape(1, sr)


@pytest.fixture
def dummy_spectrogram():
    """
    Create a dummy spectrogram with 100 frames and 40 mel bins.
    """
    return np.random.rand(100, 40).astype(np.float32)


# -------------------------------
# Tests for individual audio augment methods
# -------------------------------

def test_random_crop(dummy_audio):
    crop_length = 0.5  # seconds
    transform = RandomCrop(crop_length=crop_length, sample_rate=16000)
    output = transform(dummy_audio)
    # The cropped audio should have a number of samples equal to crop_length*sample_rate.
    assert output.shape[1] == int(crop_length * 16000)


def test_add_white_noise(dummy_audio):
    transform = AddWhiteNoise(min_amplitude=0.001, max_amplitude=0.015)
    output = transform(dummy_audio)
    assert output.shape == dummy_audio.shape
    assert not np.allclose(output, dummy_audio)


def test_add_noise_from_files(dummy_audio, tmp_path):
    # Create a temporary noise file (1 second of white noise).
    noise = np.random.randn(16000).astype(np.float32)
    noise_file = tmp_path / "noise.wav"
    sf.write(str(noise_file), noise, 16000)
    transform = AddNoiseFromFiles(noise_files=[str(noise_file)], snr_low=0, snr_high=10, sample_rate=16000, normalize=False)
    output = transform(dummy_audio)
    assert output.shape == dummy_audio.shape
    assert not np.allclose(output, dummy_audio)


def test_add_reverb_from_files(dummy_audio, tmp_path):
    # Create a temporary impulse response file (dummy IR).
    rir = np.random.randn(8000).astype(np.float32)  # 0.5 second impulse response
    rir_file = tmp_path / "rir.wav"
    sf.write(str(rir_file), rir, 16000)
    transform = AddReverbFromFiles(rir_files=[str(rir_file)], sample_rate=16000, rir_scale_factor=1.0)
    output = transform(dummy_audio)
    assert output.shape == dummy_audio.shape
    assert not np.allclose(output, dummy_audio)


def test_time_stretch(dummy_audio):
    transform = TimeStretch(min_rate=0.8, max_rate=1.25, leave_length_unchanged=True)
    output = transform(dummy_audio)
    # When leaving the length unchanged, output shape should be the same.
    assert output.shape == dummy_audio.shape


def test_pitch_shift(dummy_audio):
    transform = PitchShift(sample_rate=16000, min_semitones=-2, max_semitones=2)
    output = transform(dummy_audio)
    assert output.shape == dummy_audio.shape
    assert not np.allclose(output, dummy_audio)


def test_polarity_inversion(dummy_audio):
    transform = PolarityInversion()
    output = transform(dummy_audio)
    # The polarity inverted audio should be exactly the negative of the input.
    np.testing.assert_array_equal(output, -dummy_audio)


def test_random_gain(dummy_audio):
    transform = RandomGain(min_gain_db=-6, max_gain_db=6)
    output = transform(dummy_audio)
    assert output.shape == dummy_audio.shape
    # While it is possible for the gain to be 0, it's unlikely the output equals the input.
    assert not np.allclose(output, dummy_audio)


def test_bit_crush(dummy_audio):
    transform = BitCrush(min_bit_depth=5, max_bit_depth=8)
    output = transform(dummy_audio)
    assert output.shape == dummy_audio.shape


def test_clipping_distortion(dummy_audio):
    transform = ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=40)
    output = transform(dummy_audio)
    assert output.shape == dummy_audio.shape


def test_reverb(dummy_audio):
    transform = Reverb(
        sample_rate=16000,
        reverberance_min=0,
        reverberance_max=100,
        dumping_factor_min=0,
        dumping_factor_max=100,
        room_size_min=0,
        room_size_max=100
    )
    output = transform(dummy_audio)
    assert output.shape == dummy_audio.shape
    assert not np.allclose(output, dummy_audio)


def test_tanh_distortion(dummy_audio):
    transform = TanhDistortion(min_distortion=0.01, max_distortion=0.7)
    output = transform(dummy_audio)
    assert output.shape == dummy_audio.shape
    assert not np.allclose(output, dummy_audio)


def test_echo_effect(dummy_audio):
    echo = EchoEffect(delay=0.1, decay=0.5, sample_rate=16000)
    output = echo(dummy_audio)
    # Check shape and that echo effect changed the waveform.
    assert output.shape == dummy_audio.shape
    assert not np.allclose(output, dummy_audio)


def test_low_pass_filter(dummy_audio):
    lp_filter = LowPassFilter(cutoff_freq=1000, sample_rate=16000, order=4)
    output = lp_filter(dummy_audio)
    assert output.shape == dummy_audio.shape
    # The filtered signal should be different when high frequency is removed.
    assert not np.allclose(output, dummy_audio)


def test_high_pass_filter(dummy_audio):
    hp_filter = HighPassFilter(cutoff_freq=1000, sample_rate=16000, order=4)
    output = hp_filter(dummy_audio)
    assert output.shape == dummy_audio.shape
    # High-pass filtering should affect the waveform.
    assert not np.allclose(output, dummy_audio)


def test_band_pass_filter(dummy_audio):
    bp_filter = BandPassFilter(low_cutoff=300, high_cutoff=3400, sample_rate=16000, order=4)
    output = bp_filter(dummy_audio)
    assert output.shape == dummy_audio.shape
    # Band-pass filtered signal should be altered.
    assert not np.allclose(output, dummy_audio)


def test_reverse_audio(dummy_audio):
    reverse = ReverseAudio()
    output = reverse(dummy_audio)
    # Check that the output is the time-reversed version of the input.
    assert output.shape == dummy_audio.shape
    np.testing.assert_array_equal(output, dummy_audio[:, ::-1])


def test_dynamic_range_compression(dummy_audio):
    # Make a signal with a higher amplitude to see compression effect.
    high_amp_audio = dummy_audio * 2.0
    drc = DynamicRangeCompression(threshold=0.5, ratio=4.0)
    output = drc(high_amp_audio)
    assert output.shape == high_amp_audio.shape
    # Check that the dynamic range is reduced.
    assert np.max(np.abs(output)) < np.max(np.abs(high_amp_audio))


def test_equalizer(dummy_audio):
    # Boost frequencies around 1000Hz.
    eq = Equalizer(gains={1000: 1.0}, sample_rate=16000)
    output = eq(dummy_audio)
    assert output.shape == dummy_audio.shape
    # The equalized output should be modified.
    assert not np.allclose(output, dummy_audio)


def test_flanger(dummy_audio):
    flanger = Flanger(max_delay=0.005, depth=0.002, rate=0.25, sample_rate=16000)
    output = flanger(dummy_audio)
    assert output.shape == dummy_audio.shape
    # The flanger effect should produce a waveform different from the input.
    assert not np.allclose(output, dummy_audio)


def test_chorus(dummy_audio):
    chorus = Chorus(delays=[0.02, 0.03], decays=[0.5, 0.5], sample_rate=16000)
    output = chorus(dummy_audio)
    assert output.shape == dummy_audio.shape
    # Check that the chorus effect modifies the signal.
    assert not np.allclose(output, dummy_audio)


def test_tremolo(dummy_audio):
    tremolo = Tremolo(rate=5.0, depth=0.5, sample_rate=16000)
    output = tremolo(dummy_audio)
    assert output.shape == dummy_audio.shape
    # Tremolo modulation should change the amplitude over time.
    assert not np.allclose(output, dummy_audio)


def test_fbank(dummy_audio):
    transform = FBank(sampling_rate=16000, num_mel_bins=40, max_frame_length=100, frame_length=25, frame_shift=10)
    output = transform(dummy_audio)
    # FBank should output a spectrogram with shape (max_frame_length, num_mel_bins).
    assert output.shape == (100, 40)

def test_spec_augment(dummy_spectrogram):
    transform = SpecAugment(freq_mask_param=10, time_mask_param=20)
    output = transform(dummy_spectrogram)
    assert output.shape == dummy_spectrogram.shape

def test_cut_out(dummy_spectrogram):
    transform = CutOut(num_holes=2, length=5)
    output = transform(dummy_spectrogram)
    assert output.shape == dummy_spectrogram.shape

def test_normalize(dummy_audio):
    # Use mean=0 and std=1 for simplicity.
    transform = Normalize(mean=0.0, std=1.0)
    output = transform(dummy_audio)
    assert output.shape == dummy_audio.shape


# -------------------------------
# Tests for composite transforms
# -------------------------------

def test_random_apply(dummy_audio):
    # With p=0, the transform should not be applied.
    transform = RandomApply(AddWhiteNoise(), p=0.0)
    output = transform(dummy_audio)
    np.testing.assert_array_equal(output, dummy_audio)
    
    # With p=1, the transform should always be applied.
    transform = RandomApply(AddWhiteNoise(), p=1.0)
    output = transform(dummy_audio)
    assert not np.allclose(output, dummy_audio)


def test_random_choice_transform(dummy_audio):
    transforms = [PolarityInversion(), AddWhiteNoise()]
    transform = RandomChoiceTransform(transforms)
    output = transform(dummy_audio)
    assert output.shape == dummy_audio.shape


def test_random_order_compose(dummy_audio):
    transforms = [AddWhiteNoise(), PolarityInversion()]
    transform = RandomOrderCompose(transforms)
    output = transform(dummy_audio)
    assert output.shape == dummy_audio.shape


def test_compose(dummy_audio):
    # Two successive polarity inversions should yield the original signal.
    transforms = [PolarityInversion(), PolarityInversion()]
    transform = Compose(transforms)
    output = transform(dummy_audio)
    np.testing.assert_array_equal(output, dummy_audio)


def test_multi_view_transform(dummy_audio):
    transforms = [PolarityInversion(), AddWhiteNoise()]
    transform = MultiViewTransform(transforms)
    outputs = transform(dummy_audio)
    assert isinstance(outputs, list)
    assert len(outputs) == len(transforms)
    for out in outputs:
        assert out.shape == dummy_audio.shape


def test_to_one_hot():
    transform = ToOneHot(num_classes=10)
    labels = [1, 3, 7]
    output = transform(labels)
    expected = np.zeros((10,), dtype=float)
    expected[[1, 3, 7]] = 1
    np.testing.assert_array_equal(output, expected)
