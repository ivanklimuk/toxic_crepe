"""
Each of these variables should be imported as an env var in the future
"""

# import os

# DATALOADER PARAMETERS
LABELS_PATH = None
CATEGORICAL = False
HEADER = None
ASCII = False
RUSSIAN = True
DIGITS = True
PUNCTUATION = True
LOWER = False

# MODEL PARAMETERS
MAX_LENGTH = 231
CHANNELS = 256
KERNEL_SIZES = [15, 7, 4, 4, 3, 3]
POOLING_SIZES = [3, 3, 3]
LINEAR_SIZE = 1024
DROPOUT = 0.5
OUTPUT_SIZE = 1

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.01

MODEL_PATH = './model'
DATA_PATH = './data/tutby_full.csv'
# DATA_PATH = './data/sample.csv'

EXPERIMENT_PREFIX = 'tutby-3' + '_'

RUS = 'абвгдеёжзиклмнопрстуфхцчшщъыьэюя'

BEST_MODEL_PATH = './model/' + 'tutby-3_best.pth.tar'
