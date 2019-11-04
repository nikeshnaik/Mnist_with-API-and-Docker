from typing import Tuple, Union
import numpy as np

from digit_recognizer.models import MnistModel
import digit_recognizer.utils  as util


class DigitPredictor:

    def __init__(self):
        self.model = MnistModel()
        self.model.load_weights()


    def predict(self, image_or_filename : Union[np.ndarray,str]) -> Tuple[str,float]:
        """ Predict on a Single Image"""

        if isinstance(image_or_filename,str):
            image = util.read_image(image_or_filename,grayscale=True)

        else:
            image = image_or_filename

        return self.model.predict_on_image(image)

    def evaluate(self, dataset):
        """Evalutae on a dataset"""
        return self.model.evaluate(dataset.x_test, dataset.y_test)
