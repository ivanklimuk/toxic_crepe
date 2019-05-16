"""
Each of these variables should be imported as an env var in the future
"""

# import os

# DATALOADER PARAMETERS
MAX_LENGTH = 222
LABELS_PATH = None
CATEGORICAL = False
HEADER = None
ASCII = False
RUSSIAN = True
DIGITS = True
PUNCTUATION = True
LOWER = False

# MODEL PARAMETERS
CHANNELS = 256
KERNEL_SIZES = [7, 7, 4, 4, 3, 3]
POOLING_SIZE = 3
LINEAR_SIZE = 1024
DROPOUT = 0.35
OUTPUT_SIZE = 1

EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.01

MODEL_PATH = './model'
DATA_PATH = './data/tutby_short.csv'
# DATA_PATH = './data/sample.csv'

EXPERIMENT_PREFIX = 'tutby-2' + '_'

RUS = 'абвгдеёжзиклмнопрстуфхцчшщъыьэюя'

BEST_MODEL_PATH = './model/' + 'tutby-2_best.pth.tar'
