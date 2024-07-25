from typing import Optional, Union, List

import torch
import numpy as np
from transformers.utils import TensorType
from transformers.feature_extraction_utils import BatchFeature

from audio_augment.utils import to_tensor


class HuggingFaceAudioDatasetCollator(object):

    def __init__(
        self, 
        do_normalize: bool = True, 
        mean: float = -4.2677393, 
        std: float = 4.5689974, 
    ):
        self.do_normalize = do_normalize
        self.mean = mean
        self.std = std

    def __call__(
        self, 
        examples, 
        return_tensors: Optional[Union[str, TensorType]] = 'pt',
    ) -> BatchFeature:
        features = [example['input_values'] for example in examples]
        padded_inputs = BatchFeature({"input_values": features})

        input_values = padded_inputs.get("input_values")
        if isinstance(input_values[0], list):
            padded_inputs["input_values"] = [np.asarray(feature, dtype=np.float32) for feature in input_values]
        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)
        padded_inputs['labels'] = torch.vstack(
            [to_tensor(example['labels'], device='cpu') for example in examples]
        ).float()

        return padded_inputs