import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

import config
import pickle
from janome.analyzer import Analyzer
from janome.tokenizer import Tokenizer as JPTKZ
from vws import RDRSegmenter as RDR, Tokenizer as VNTKZ

rdr = RDR.RDRSegmenter()
vntk = VNTKZ.Tokenizer()

jptk = JPTKZ()
analyzer = Analyzer(char_filters=config.JP_CHAR_FILTERS, tokenizer=jptk, token_filters=config.TOKEN_FILTERS)

class Vocab:
    def __init__(self, name:str, use_jp:bool=False, use_vi:bool=False):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1}
        self.word2count = {"SOS": 0, "EOS": 0}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2
        if use_jp:
            if use_vi:
                ValueError("Can't use multiple tokenizers!")
        self.use_jp = use_jp
        self.use_vi = use_vi

    def add_sentence(self, sentence:str):
        words = word_segment(self, sentence)
        for w in words:
            self.add_word(w)

    def add_word(self, word:str):
        if word in self.word2count:
            self.word2count[word] += 1
        else:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1

    def remove_word(self, word:str):
        if word != "SOS" and word != "EOS":
            self.n_words -= 1
            idx = self.word2index[word]
            last_word = self.index2word[self.n_words]

            del self.word2index[word]
            del self.word2count[word]
            del self.index2word[self.n_words]

            if last_word != word:
                self.word2index[last_word] = idx
                self.index2word[idx] = last_word
    
    def save(self, path:str):
        with open(path+'/'+self.name+'.pk', 'wb') as f:
            pickle.dump(self, f)
            f.close()

    def __len__(self):
        return self.n_words

class VocabDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i:int):
        return self.pairs[i]

class Collate:
    def __init__(self, input_lang:Vocab, output_lang:Vocab, max_length:int=30):
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.max_length = max_length
    def __call__(self, batch):
        batch_size = len(batch)
        input_tensor = torch.Tensor(batch_size, self.max_length)
        output_tensor = torch.Tensor(batch_size, self.max_length)
        for i in range(batch_size):
            input_tensor[i] = sen2idx(self.input_lang, batch[i][0], self.max_length)
            output_tensor[i] = sen2idx(self.output_lang, batch[i][1], self.max_length)
        return Variable(input_tensor), Variable(output_tensor)

def _drop_filters(sentence:str):
    return ''.join([char for char in sentence if char not in config.FILTERS]).strip().lower()

def word_segment(lang:Vocab, sentence:str):
    words = None
    if lang.use_jp:
        norm_sen = ''.join([w for w in analyzer.analyze(sentence)])
        words = [w for w in analyzer.analyze(_drop_filters(norm_sen))]
    elif lang.use_vi:
        norm_sen = _drop_filters(sentence)
        words = [w for w in rdr.segmentRawSentences(vntk, norm_sen).split(' ')]
    else:
        words = [w for w in _drop_filters(sentence).split(' ')]
    return words

def sen2idx(lang:Vocab, sentence:str, max_length:int=30):
    words = word_segment(lang, sentence)
    indexes = [lang.word2index[word] for word in words]
    result = torch.Tensor(max_length)
    result[:] = config.EOS_token
    for i, index in enumerate(indexes):
        result[i] = index
    return result