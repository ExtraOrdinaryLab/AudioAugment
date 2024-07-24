from typing import Dict, Optional, Callable

from torch import Tensor
from datasets import load_dataset
from datasets.features import Audio
from torch.utils.data import Dataset as TorchDataset


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

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
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