# Audio-Augment

A collection of audio augmentation transforms implemented in Python. This repository provides various methods to manipulate and enhance audio data for robust machine learning applications such as speech recognition and audio classification.

## Installation

Ensure you have Python 3.9+ installed. Install the required packages:

```bash
pip install git+https://github.com/ExtraOrdinaryLab/AudioAugment.git
```

## Usage

Import and use any of the transforms in your project:

```python
from audio_augment import AddWhiteNoise, RandomCrop, Compose

# Example: Compose a pipeline
pipeline = Compose([
    RandomCrop(crop_length=0.5, sample_rate=16000),
    AddWhiteNoise(min_amplitude=0.001, max_amplitude=0.015)
])

augmented_audio = pipeline(audio_data)
```

## License

This project is licensed under the MIT License.

Enjoy augmenting your audio data!