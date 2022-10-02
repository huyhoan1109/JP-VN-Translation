import json
import torch
import string
import pickle
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
INIT_VOCAB = 'Tatoeba_2K'
DATA_DIR = './data'
VOCAB_DIR = './data/vocab/'
TRAIN_DIR = './data/train/'
TEST_DIR = './data/test/'
LOADER_DIR = './loader/'
USE_PERCENT = 80
N_WORKERS = 0 # if you have GPU then change as you like
JP_CHAR_FILTERS = [UnicodeNormalizeCharFilter()]
TOKEN_FILTERS = [POSStopFilter(['記号','助詞']), LowerCaseFilter(), ExtractAttributeFilter('surface')]
FILTERS = list(string.punctuation) + list(string.digits) + list('…')

def read_file(root, format=None):
    data = []
    try:
        if format == '.json': 
            with open(root, 'r', encoding='utf-8') as f:
                translate = json.load(f)
            for i in translate:
                data.append([str(i), str(translate[i])])
        if format in ['.pk', 'pickle']:
            with open(root, 'rb', encoding='utf-8') as f:
                data = pickle.load(f)
        else:
            with open(root+'/data-ja.txt', 'r', encoding='utf-8') as f:
                jp_lines = f.read().split('\n')
            with open(root+'/data-vi.txt', 'r', encoding='utf-8') as f:
                vi_lines = f.read().split('\n')
            for jp, vi in zip(jp_lines, vi_lines):
                data.append([jp, vi])
    except IOError:
        print("An IOError has occured!")
    return data
# data = read_file(ORIGINAL_DS_DIR+INIT_VOCAB)
# print(data)