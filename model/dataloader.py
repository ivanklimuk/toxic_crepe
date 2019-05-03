import numpy as np
import pandas as pd
import string
import torch
from sklearn.model_selection import train_test_split
from constants import RUS


class DataLoader(object):
    def __init__(self,
                 data_path=None,
                 labels_path=None,
                 categorical=False,
                 max_length=222,
                 header=None,
                 ascii=False,
                 russian=True,
                 digits=True,
                 punctuation=True,
                 lower=False
                 ):
        self.data_path = data_path
        self.labels_path = labels_path
        self.categorical = categorical
        self.max_length = max_length
        self.header = header
        self.ascii = ascii
        self.russian = russian
        self.digits = digits
        self.punctuation = punctuation
        self.lower = lower

    def _load_data(self):
        data = np.array(pd.read_csv(self.data_path, header=self.header))
        if self.labels_path:
            labels = np.array(pd.read_csv(self.labels_path, header=self.header))
        else:
            labels, data = data[:, 0], data[:, 1]
        
        if self.categorical:
            labels = np.vstack((
                (labels == 1).astype(int),
                (labels == 0).astype(int)
            )).T
        else:
            labels = labels.reshape(-1, 1)

        self.text_data = data
        self.labels = labels

    def create_vocabulary(self):
        alphabet = []
        if self.ascii:
            alphabet += list(string.ascii_lowercase)
        if self.digits:
            alphabet += list(string.digits)
        if self.punctuation:
            alphabet += list(string.punctuation) + ['\n', ' ']
        if self.russian:
            alphabet += list(RUS + RUS.upper())
        #self.alphabet = set(alphabet)
        self.alphabet = alphabet
        self.vocabulary_size = len(self.alphabet)
        vocabulary = {character: number + 1 for number, character in enumerate(self.alphabet)}
        self.vocabulary = vocabulary

    def text_to_array(self, texts):
        array = np.zeros((len(texts), self.max_length), dtype=np.int)
        for row, line in enumerate(texts):
            if self.lower:
                line = line.lower()
            for column in range(min([len(line), self.max_length])):
                array[row, column] = self.vocabulary.get(line[column], 0)  # replace with 0
        return array

    def load_data(self, split=True):
        self._load_data()
        self.create_vocabulary()
        self.data = self.text_to_array(self.text_data)

        self.data = self.data.astype(np.float)
        self.labels = self.labels.astype(np.float)

        if split:
            return train_test_split(self.data, self.labels, random_state=128)
        else:
            return self.data, self.labels


def data_iterator(data, labels, batch_size, shuffle=True):
    while True:
        if shuffle:
            shuf = np.random.permutation(len(data))
            data = data[shuf]
            labels = labels[shuf]
        for i in range(0, len(data), batch_size):
            yield torch.autograd.Variable(torch.from_numpy(data[i:i + batch_size]).float()), \
                  torch.autograd.Variable(torch.from_numpy(labels[i:i + batch_size]).float())
