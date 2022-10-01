
import torch
import string
from janome.charfilter import *
from janome.tokenfilter import *

MAX_LEN = 20
BATCH_SIZE = 256
TEST_SIZE = 0.2
TEACHER_FORCING_RATIO = 0.2
MAX_LENGTH = 30
LR = 0.001
HIDDEN_SIZE = 300
DIRECTIONS = 2
LAYERS = 1
DROPOUT = 0.15

SOS_token = 0
EOS_token = 1
MIN_VOCAB_WORDS = 2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ORIGINAL_DS_DIR = './VN-JP-NLP-Dataset/'
INIT_VOCAB_DIR = './VN-JP-NLP-Dataset/Glosbe282K'
VOCAB_DIR = './data/t'
TRAIN_DIR = './data/train'
TEST_DIR = './data/test'
USE_PERCENT = 80
JP_CHAR_FILTERS = [UnicodeNormalizeCharFilter()]
TOKEN_FILTERS = [POSStopFilter(['記号','助詞']), LowerCaseFilter(), ExtractAttributeFilter('surface')]
FILTERS = list(string.punctuation) + list(string.digits)