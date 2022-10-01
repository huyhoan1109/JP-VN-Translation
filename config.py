import torch
import string
from janome.charfilter import *
from janome.tokenfilter import *

BATCH_SIZE = 100
TEST_SIZE = 0.2
WEIGTH_DECAY = 0.2
MAX_LENGTH = 30
LEARNING_RATE = 0.001
HIDDEN_SIZE = 300
DIRECTIONS = 2
DROPOUT = 0.15

SOS_token = 0
EOS_token = 1
MIN_VOCAB_WORDS = 2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ORIGINAL_DS_DIR = './VN-JP-NLP-Dataset/'
INIT_VOCAB = 'wiki_20K'
VOCAB_DIR = './data/vocab'
TRAIN_DIR = './data/train'
TEST_DIR = './data/test'
USE_PERCENT = 80
N_WORKERS = 0 # if you have GPU then change as you like
JP_CHAR_FILTERS = [UnicodeNormalizeCharFilter()]
TOKEN_FILTERS = [POSStopFilter(['記号','助詞']), LowerCaseFilter(), ExtractAttributeFilter('surface')]
FILTERS = list(string.punctuation) + list(string.digits)