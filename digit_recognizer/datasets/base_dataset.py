from pathlib import Path
import argparse
import os
from digit_recognizer import utils

class Dataset:

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[2] / 'data'

    def load_or_generate_data(self):
        pass


def _download_raw_dataset(metadata):
    if os.path.exists(metadata['filename']):
        return
    print('Downloading raw dataset...')
    utils.download_url(metadata['url'], metadata['filename'])
    # print('Computing SHA-256...')l
    # sha256 = utils.compute_sha256(metadata['filename'])
    # if sha256 != metadata['sha256']:
    #     raise ValueError('Downloaded data file SHA-256 does not match that listed in metadata document.')


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample_fraction",
                        type=float,
                        default=None,
                        help="If given, is used as the fraction of data to expose.")
    return parser.parse_args()
