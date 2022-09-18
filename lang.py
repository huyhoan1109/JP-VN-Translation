import config
import pickle
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

from janome.analyzer import Analyzer
from janome.tokenizer import Tokenizer as JPTKZ
from vws import RDRSegmenter as RDR, Tokenizer as VNTKZ

rdr = RDR.RDRSegmenter()
vntk = VNTKZ.Tokenizer()

jptk = JPTKZ()
analyzer = Analyzer(char_filters=config.JP_CHAR_FILTERS, tokenizer=jptk, token_filters=config.TOKEN_FILTERS)

class Vocab:
    """
     Holds vocabulary for each language and dictionaries to convert words to and from indexes
    """
    def __init__(self, name, use_jp=False, use_vi=False):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1}
        self.word2count = {"SOS": 0, "EOS": 0}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2
        self.use_jp = use_jp
        self.use_vi = use_vi
    
    def _drop_filters(self, sentence):
        return ''.join([char for char in sentence if char not in config.FILTERS]).strip().lower()

    def add_sentence(self, sentence):
        if not self.use_jp and not self.use_vi:
            for word in sentence.split(" "):
                self.add_word(word)
        elif self.use_jp:
            sen = ''.join([w for w in analyzer.analyze(sentence)])
            [self.add_word(w) for w in analyzer.analyze(self._drop_filters(sen))]
        elif self.use_vi:
            sen = self._drop_filters(sentence)
            [self.add_word(w) for w in rdr.segmentRawSentences(vntk, sen).split(' ')]
    
    def add_word(self, word):
        if word in self.word2count:
            self.word2count[word] += 1
        else:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1

    def remove_word(self, word):
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
    
    def save(self, path):
        with open(path+'/'+self.name+'.pk', 'wb') as f:
            pickle.dump(self, f)
            f.close()

    def __len__(self):
        return self.n_words

class VocabDataset(Dataset):
    def __init__(self, input_vocab, output_vocab):
        self.input_vocab = input_vocab 
        self.output_vocab = output_vocab
    
    def __len__(self):
        return len(self.input_vocab)

    def __getitem__(self):
        return len()