# AudioAugment

# Usage

```python
from torch.utils.data import DataLoader

from audio_augment.datasets import AudiosetDataset
from audio_augment.transforms import (
    RandomCrop, 
    AddWhiteNoise, 
    TimeStretch, 
    PitchShift, 
    PolarityInversion, 
    RandomGain, 
    FBank, 
    RandomApply, 
    Compose, 
    RandomOrderCompose, 
    ToOneHot
)
from audio_augment.collators import HuggingFaceAudioDatasetCollator


def main():
    data_dir = '/mnt/data1_HDD_4TB/yang/corpus/audioset'
    
    train_dataset = AudiosetDataset(
        dataset_name='confit/audioset', 
        dataset_config_name='20k', 
        data_dir=data_dir, 
        split='train', 
        sampling_rate=16000, 
        num_classes=527, 
        audio_column_name='audio', 
        label_column_name='label', 
        trust_remote_code=True, 
        transform=Compose([
            RandomCrop(crop_length=10, sample_rate=16000), 
            RandomOrderCompose([
                RandomApply([AddWhiteNoise(min_amplitude=0.001, max_amplitude=0.015)], p=0.5), 
                RandomApply([TimeStretch(min_rate=0.8, max_rate=1.25)], p=0.5), 
                RandomApply([PitchShift(sample_rate=16000, min_semitones=-2, max_semitones=2)], p=0.5), 
                RandomApply([PolarityInversion()], p=0.5), 
                RandomApply([RandomGain(min_gain_db=0, max_gain_db=10)], p=0.5), 
            ]), 
            FBank(sampling_rate=16000, num_mel_bins=128, max_frame_length=1024)
        ]), 
        target_transform=ToOneHot(num_classes=527)
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=False, 
        collate_fn=HuggingFaceAudioDatasetCollator(sampling_rate=16000)
    )
    batch = next(iter(train_dataloader))


if __name__ == '__main__':
    main()
```