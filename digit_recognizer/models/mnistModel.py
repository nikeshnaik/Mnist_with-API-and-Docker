from typing import Callable, Dict, Tuple

import numpy as np

from digit_recognizer.models.base_model import Model
from digit_recognizer.datasets.mnist_dataset import MNISTDataset
from digit_recognizer.networks.mlp import mlp


class MnistModel(Model):

    def __init__(self,
                 dataset_cls: type = MNISTDataset,
                 network_fn: Callable = mlp,
                 dataset_args: Dict = None,
                 network_args: Dict = None):
        """Define the default dataset and network values for this model."""
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)


    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)
        # NOTE: integer to character mapping dictionary is self.data.mapping[integer]
        # Your code below (Lab 1)
        import pdb; pdb.set_trace()
        image = image.reshape(-1)
        pred_raw = self.network.predict(np.expand_dims(image,0),batch_size=1).flatten()
        ind = np.argmax(pred_raw)
        confidence_of_prediction = pred_raw[ind]
        predicted_character = self.data.mapping[ind]
        # Your code above (Lab 1)
        return predicted_character, confidence_of_prediction
