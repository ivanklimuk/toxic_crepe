import os

MAX_LENGTH = 222

# MODEL PARAMETERS
CHANNELS=256
KERNEL_SIZES=[7, 7, 4, 4, 3, 3]
POOLING_SIZE=3
LINEAR_SIZE=1024
DROPOUT=0.35
OUTPUT_SIZE=1

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001

MODEL_PATH = './model'
DATA_PATH = './data/tutby_short.csv'
#DATA_PATH = './data/sample.csv'

EXPERIMENT_PREFIX = 'tutby-1_'

RUS = 'абвгдеёжзиклмнопрстуфхцчшщъыьэюя'

BEST_MODEL_PATH = './model/' + 'tutby-1_best.pth.tar'
