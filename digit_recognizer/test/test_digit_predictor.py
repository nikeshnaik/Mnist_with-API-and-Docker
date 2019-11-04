import os
from pathlib import Path
import unittest

from digit_recognizer.digit_predictor import DigitPredictor

SUPPORT_DIRNAME = Path(__file__).parents[0].resolve()/'support'/'mnist'


class TestDigitPredictor(unittest.TestCase):

    def test_filename(self):
        predictor = DigitPredictor()

        for filename in SUPPORT_DIRNAME.glob("*.png"):
            pred, conf = predictor.predict(str(filename))
            print(f'Prediction: {pred} at confidence: {conf} for image with digit {filename,stem}')
            self.assertEqual(pred, filename.stem)
            self.assertGreater(conf, 0.7)
