import json
import os
from pathlib import Path
import shutil
import zipfile

from boltons.cacheutils import cachedproperty
from tensorflow.keras.utils import to_categorical
import h5py
import numpy as np
import toml
from digit_recognizer.datasets.base_dataset import _download_raw_dataset, Dataset, _parse_args


SAMPLE_TO_BALANCE = False

RAW_DATA_DIRNAME = Dataset.data_dirname() / 'raw' / 'mnist'
METADATA_FILENAME = RAW_DATA_DIRNAME / 'metadata.toml'

PROCESSED_DATA_DIRNAME = Dataset.data_dirname()/'processed'/'mnist'
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / 'byclass.h5'

ESSENTIALS_FILENAME = Path(__file__).parents[0].resolve() / 'mnist_essentials.json'


class MNISTDataset(Dataset):

    def __init__(self, subsample_fraction: float = None):
        if not os.path.exists(ESSENTIALS_FILENAME):
            _download_and_process_emnist()
        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)

        self.mapping = _augment_emnist_mapping(dict(essentials['mapping']))
        self.num_classes = essentials['num_classes']
        self.input_shape = essentials['input_shape']
        self.output_shape = (self.num_classes,)

        self.subsample_fraction = subsample_fraction
        self.x_train = None
        self.y_train_int = None
        self.x_test = None
        self.y_test_int = None

    def load_or_generate_data(self):
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_and_process_emnist()
        with h5py.File(PROCESSED_DATA_FILENAME, 'r') as f:
            self.x_train = f['x_train'][:]
            self.y_train_int = f['y_train'][:]
            self.x_test = f['x_test'][:]
            self.y_test_int = f['y_test'][:]
        self._subsample()

    def _subsample(self):
        """Only this fraction of data will be loaded."""
        if self.subsample_fraction is None:
            return
        num_train = int(self.x_train.shape[0] * self.subsample_fraction)
        num_test = int(self.x_test.shape[0] * self.subsample_fraction)
        self.x_train = self.x_train[:num_train]
        self.y_train_int = self.y_train_int[:num_train]
        self.x_test = self.x_test[:num_test]
        self.y_test_int = self.y_test_int[:num_test]

    @cachedproperty
    def y_train(self):
        return to_categorical(self.y_train_int, self.num_classes)

    @cachedproperty
    def y_test(self):
        return to_categorical(self.y_test_int, self.num_classes)

    def __repr__(self):
        return (
            'MNIST Dataset\n'
            f'Num classes: {self.num_classes}\n'
            f'Input shape: {self.input_shape}\n'
        )


def _download_and_process_emnist():
    metadata = toml.load(METADATA_FILENAME)
    curdir = os.getcwd()
    os.chdir(RAW_DATA_DIRNAME)
    _download_raw_dataset(metadata)
    _process_raw_dataset(metadata['filename'])
    os.chdir(curdir)


def _process_raw_dataset(filename: str):
    """ Unzip and Process Raw Dataset After Download """
    print('Unzipping MNIST...')
    zip_file = zipfile.ZipFile(filename, 'r')
    zip_file.extract('mnist.csv')

    print("Loading Data from CSV File")
    from pandas import read_csv
    from sklearn.model_selection import train_test_split

    data = read_csv('mnist.csv')
    X = data.loc[:,:'col_784']
    Y = data.loc[:,'y']
    x_train,x_test, y_train,y_test = train_test_split(X,Y,test_size=0.3)


    if SAMPLE_TO_BALANCE:
        print('Balancing classes to reduce amount of data')
        x_train, y_train = _sample_to_balance(x_train, y_train)
        x_test, y_test = _sample_to_balance(x_test, y_test)


    print('Saving to HDF5 in a compressed format...')
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    with h5py.File(PROCESSED_DATA_FILENAME, 'w') as f:
        f.create_dataset('x_train', data=x_train, dtype='u1', compression='lzf')
        f.create_dataset('y_train', data=y_train, dtype='u1', compression='lzf')
        f.create_dataset('x_test', data=x_test, dtype='u1', compression='lzf')
        f.create_dataset('y_test', data=y_test, dtype='u1', compression='lzf')

    print('Saving essential dataset parameters to digit_recognizer/datasets...')
    num_classes = len(set(data['y']))
    essentials = {'input_shape': list(x_train.shape[1:]),'num_classes':num_classes}
    with open(ESSENTIALS_FILENAME, 'w') as f:
        json.dump(essentials, f)

    print('Cleaning up...')
    print("Current dir -->",os.getcwd())
    os.remove('mnist.zip')
    os.remove('mnist.csv')



def _sample_to_balance(x, y):
    """Because the dataset is not balanced, we take at most the mean number of instances per class."""
    np.random.seed(42)
    num_to_sample = int(np.bincount(y.flatten()).mean())
    all_sampled_inds = []
    for label in np.unique(y.flatten()):
        inds = np.where(y == label)[0]
        sampled_inds = np.unique(np.random.choice(inds, num_to_sample))
        all_sampled_inds.append(sampled_inds)
    ind = np.concatenate(all_sampled_inds)
    x_sampled = x[ind]
    y_sampled = y[ind]
    return x_sampled, y_sampled


def _augment_emnist_mapping(mapping):
    """Augment the mapping with extra symbols."""
    # Extra symbols in IAM dataset
    # extra_symbols = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '?']

    # padding symbol
    # extra_symbols.append('_')

    # max_key = max(mapping.keys())
    # extra_mapping = {}
    # for i, symbol in enumerate(extra_symbols):
    #     extra_mapping[max_key + 1 + i] = symbol
    #
    # return {**mapping, **extra_mapping}
    return mapping



def main():
    """Load EMNIST dataset and print info."""
    args = _parse_args()
    dataset = MNISTDataset(subsample_fraction=args.subsample_fraction)
    dataset.load_or_generate_data()

    print(dataset)
    print(dataset.x_train.shape, dataset.y_train.shape)  # pylint: disable=E1101
    print(dataset.x_test.shape, dataset.y_test.shape)  # pylint: disable=E1101


if __name__ == '__main__':
    main()
