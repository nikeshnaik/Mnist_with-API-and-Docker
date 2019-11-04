import os
from pathlib import Path
from time import time
import unittest

from digit_recognizer.datasets.mnist_dataset import MNIST
from digit_recognizer.digit_predictor import DigitPredictor

os.environ["CUDA_VISIBLE_DEVICES"] = ""

SUPPORT_DIRNAME = Path(__file__).parents[0].resolve()/'support'/'emnist'


class TestEvaluateDigitPredictor(unittest.TestCase):

    def test_evalaute(self):

        predictor = DigitPredictor()
        dataset = MNIST()
        dataset.load_or_generate_data()
        t = time()
        metric = predictor.evaluate(dataset)
        time_taken = time() - t
        print(f'acc: {metric}, time taken: {time_taken}')
        self.assertGreater(metrix,0.6)
        self.assertLess(time_taken, 10)
