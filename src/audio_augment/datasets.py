import random
from copy import deepcopy
from typing import Dict, Optional, Callable

import torch
import numpy as np
from torch import Tensor
from datasets import load_dataset
from datasets.features import Audio
from torch.utils.data import Dataset as TorchDataset

from audio_augment.transforms import ToOneHot
from audio_augment.utils import to_tensor, to_numpy


class AudiosetDataset(TorchDataset):

    def __init__(
        self, 
        dataset_name: str = 'confit/audioset', 
        dataset_config_name: str = '20k', 
        data_dir: str = None, 
        split: str = 'train', 
        sampling_rate: int = 16000, 
        num_classes: int = 527, 
        audio_column_name: str = 'audio', 
        label_column_name: str = 'label', 
        trust_remote_code: bool = True, 
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.split = split
        self.sampling_rate = sampling_rate
        self.audio_column_name = audio_column_name
        self.label_column_name = label_column_name
        self.num_classes = num_classes

        raw_datasets = load_dataset(
            dataset_name, 
            dataset_config_name, 
            data_dir=data_dir, 
            split=split, 
            trust_remote_code=trust_remote_code
        )
        self.dataset = raw_datasets.cast_column(
            audio_column_name, Audio(sampling_rate=sampling_rate)
        )

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        if len(self.dataset[index][self.audio_column_name]['array'].flatten()) == 0:
            return self.__getitem__(index + 1)

        example = self.dataset[index]
        input_values = example[self.audio_column_name]['array']
        target = example[self.label_column_name]

        if self.transform is not None:
            input_values = self.transform(input_values)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'input_values': input_values, 'labels': target}


class AudiosetDatasetWithMixup(AudiosetDataset):

    def __init__(
        self, 
        dataset_name: str = 'confit/audioset', 
        dataset_config_name: str = '20k', 
        data_dir: str = None, 
        split: str = 'train', 
        sampling_rate: int = 16000, 
        num_classes: int = 527, 
        mixup: bool = True, 
        mixup_rate: float = 0.5, 
        audio_column_name: str = 'audio', 
        label_column_name: str = 'label', 
        trust_remote_code: bool = True, 
        transform: Optional[Callable] = None, 
        transform_after_mixup: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None, 
    ) -> None:
        self.mixup = mixup
        self.mixup_rate = mixup_rate
        self.transform_after_mixup = transform_after_mixup
        super().__init__(
            dataset_name, 
            dataset_config_name, 
            data_dir, split, 
            sampling_rate, 
            num_classes, 
            audio_column_name, 
            label_column_name, 
            trust_remote_code, 
            transform, 
            target_transform
        )

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        if len(self.dataset[index][self.audio_column_name]['array'].flatten()) == 0:
            return self.__getitem__(index + 1)

        example = self.dataset[index]
        input_values = example[self.audio_column_name]['array']
        target = example[self.label_column_name]

        if self.transform is not None:
            input_values = self.transform(input_values)

        if self.target_transform is not None:
            target = self.target_transform(target)

        onehot = ToOneHot(self.num_classes)
        target = onehot(target)

        if self.mixup and (random.random() < self.mixup_rate):
            random_indices = list(range(len(self.dataset)))
            random_indices.remove(index)
            mixup_idx = random.choice(random_indices)
            mixup_audio = self.dataset[mixup_idx][self.audio_column_name]['array']
            mixup_label = self.dataset[mixup_idx][self.label_column_name]

            if self.transform is not None:
                mixup_audio = self.transform(mixup_audio)

            # Apply Mixup
            lam = np.random.beta(10, 10)
            input_values = input_values - input_values.mean()
            mixup_audio = mixup_audio - mixup_audio.mean()
            # Ensure the waveforms have the same length by padding or truncating if necessary
            if input_values.shape[1] != mixup_audio.shape[1]:
                if input_values.shape[1] > mixup_audio.shape[1]:
                    # Pad the second waveform with zeros to match the length of the first waveform
                    waveform_tmp = np.zeros_like(input_values)
                    waveform_tmp[:, 0:mixup_audio.shape[1]] = mixup_audio
                    mixup_audio = deepcopy(waveform_tmp)
                else:
                    # Truncate the second waveform to match the length of the first waveform
                    mixup_audio = mixup_audio[:, 0:input_values.shape[1]]
            input_values = lam * input_values + (1 - lam) * mixup_audio
            input_values = input_values - input_values.mean()

            # Apply Mixup for labels
            mixup_label = onehot(mixup_label)
            target = lam * target + (1 - lam) * mixup_label

        if self.transform_after_mixup is not None:
            input_values = self.transform_after_mixup(input_values)
                
        return {'input_values': input_values, 'labels': target}