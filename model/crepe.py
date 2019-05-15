import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, \
                            precision_score, \
                            recall_score, \
                            fbeta_score, \
                            roc_auc_score, \
                            confusion_matrix as cm


class Crepe(nn.Module):
    def __init__(self,
                 vocabulary_size,
                 channels,
                 kernel_sizes,
                 pooling_size,
                 linear_size,
                 dropout,
                 output_size
                 ):
        super(Crepe, self).__init__()

        # define theparameters
        self.vocabulary_size = vocabulary_size
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.pooling_size = pooling_size
        self.linear_size = linear_size
        self.dropout = dropout
        self.output_size = output_size

        # define the model layers
        self.convolution_0 = nn.Conv1d(in_channels=self.vocabulary_size,
                                       out_channels=self.channels,
                                       kernel_size=self.kernel_sizes[0])
        nn.init.normal_(self.convolution_0.weight, mean=0, std=0.1)
        self.max_pooling_0 = nn.MaxPool1d(kernel_size=self.pooling_size)
        self.activation_0 = nn.ReLU(inplace=False)

        self.convolution_1 = nn.Conv1d(in_channels=self.channels,
                                       out_channels=self.channels,
                                       kernel_size=self.kernel_sizes[1])
        nn.init.normal_(self.convolution_1.weight, mean=0, std=0.1)
        self.max_pooling_1 = nn.MaxPool1d(kernel_size=self.pooling_size)
        self.activation_1 = nn.ReLU(inplace=False)

        self.convolution_2 = nn.Conv1d(in_channels=self.channels,
                                       out_channels=self.channels,
                                       kernel_size=self.kernel_sizes[2])
        nn.init.normal_(self.convolution_2.weight, mean=0, std=0.1)
        self.activation_2 = nn.ReLU(inplace=False)

        self.convolution_3 = nn.Conv1d(in_channels=self.channels,
                                       out_channels=self.channels,
                                       kernel_size=self.kernel_sizes[3])
        nn.init.normal_(self.convolution_3.weight, mean=0, std=0.1)
        self.activation_3 = nn.ReLU(inplace=False)

        self.convolution_4 = nn.Conv1d(in_channels=self.channels,
                                       out_channels=self.channels,
                                       kernel_size=self.kernel_sizes[4])
        nn.init.normal_(self.convolution_4.weight, mean=0, std=0.1)
        self.activation_4 = nn.ReLU(inplace=False)

        self.convolution_5 = nn.Conv1d(in_channels=self.channels,
                                       out_channels=self.channels,
                                       kernel_size=self.kernel_sizes[5])
        nn.init.normal_(self.convolution_1.weight, mean=0, std=0.1)
        self.max_pooling_5 = nn.MaxPool1d(kernel_size=self.pooling_size)
        self.activation_5 = nn.ReLU(inplace=False)

        self.linear_6 = nn.Linear(self.linear_size, self.linear_size)
        self.batch_norm_6 = nn.BatchNorm1d(self.linear_size)
        self.activation_6 = nn.ReLU(inplace=False)
        self.drop_out_6 = nn.Dropout(p=self.dropout, inplace=False)

        self.linear_7 = nn.Linear(self.linear_size, self.linear_size)
        self.batch_norm_7 = nn.BatchNorm1d(self.linear_size)
        self.activation_7 = nn.ReLU(inplace=False)
        self.drop_out_7 = nn.Dropout(p=self.dropout, inplace=False)

        self.linear_output = nn.Linear(self.linear_size, self.output_size)
        self.activation_output = nn.Sigmoid()

    def one_hot(self, x):
        x = x.numpy().astype(int)
        one_hot_encoded = np.zeros((x.shape[0], self.vocabulary_size, x.shape[1]))

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                one_hot_encoded[i, x[i, j], j] = 1 if x[i, j] >= 0 else 0

        return torch.FloatTensor(one_hot_encoded)

    def forward(self, s):
        # one hot encoding
        s = self.one_hot(s)
        # convolution x 6
        s = self.activation_0(self.max_pooling_0(self.convolution_0(s)))
        s = self.activation_1(self.max_pooling_1(self.convolution_1(s)))
        s = self.activation_2(self.convolution_2(s))
        s = self.activation_3(self.convolution_3(s))
        s = self.activation_4(self.convolution_4(s))
        s = self.activation_5(self.max_pooling_5(self.convolution_5(s)))

        # flatten before the FC layer
        s = s.view(s.size(0), -1)
        # linear + batchnorm + dropout x 2
        s = self.drop_out_6(self.activation_6(self.batch_norm_6(self.linear_6(s))))
        s = self.drop_out_7(self.activation_7(self.batch_norm_7(self.linear_7(s))))
        # sigmoid output
        s = self.activation_output(self.linear_output(s))

        return s


def accuracy(y_true, y_predicted):
    return accuracy_score(y_true, np.round(y_predicted))


def precision(y_true, y_predicted):
    return precision_score(y_true, np.round(y_predicted), average='macro')


def recall(y_true, y_predicted):
    return recall_score(y_true, np.round(y_predicted), average='macro')


def f1(y_true, y_predicted):
    return fbeta_score(y_true, np.round(y_predicted), average='macro', beta=1)


def confusion_matrix(y_true, y_predicted):
    return cm(y_true, np.round(y_predicted))


def roc(y_true, y_predicted):
    return roc_auc_score(y_true, y_predicted)


metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1
}
