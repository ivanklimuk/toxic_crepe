"""
Temporary solution: heavy refactoring needed!
"""

from model.dataloader import DataLoader
from model.crepe import Crepe
from constants import *
import torch


def load_model(best_model_path):
    data_loader = DataLoader(DATA_PATH,
                             labels_path=None,
                             categorical=False,
                             max_length=MAX_LENGTH,
                             header=None,
                             ascii=False,
                             russian=True,
                             digits=True,
                             punctuation=True,
                             lower=False)
    data_loader.create_vocabulary()
    model = Crepe(vocabulary_size=data_loader.vocabulary_size,
                  channels=CHANNELS,
                  kernel_sizes=KERNEL_SIZES,
                  pooling_sizes=POOLING_SIZES,
                  linear_size=LINEAR_SIZE,
                  dropout=DROPOUT,
                  output_size=OUTPUT_SIZE)
    checkpoint = torch.load(best_model_path)
    model_state = checkpoint['state_dict']
    model.load_state_dict(model_state)
    model.eval()

    return model, data_loader


def predict(text, model, data_loader):
    data = data_loader.text_to_array(text)
    prediction = model(torch.from_numpy(data))

    return prediction.data.view(-1).tolist()
