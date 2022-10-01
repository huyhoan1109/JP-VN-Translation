
import vocab
import config

import pickle
from torch.utils.data import DataLoader

def get_loader(
    root,
    vocab_dir:str=config.VOCAB_DIR,
    batch_size:int=1,
    shuffle:bool=True,
    num_workers:int=8,
    pin_memory:bool=True,
    train=True
):
    j2v = {
        {'私は': 'tôi'},
        {'君': 'cậu'},
    }
    # j2v: num_sentence, 2
    with open(f"{vocab_dir}/ja.pk", "rb") as f:
        lang1 = pickle.load(f)
    with open(f"{vocab_dir}/vi.pk", "rb") as f:
        lang2 = pickle.load(f)
    dataset = vocab.VocabDataset(j2v)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=vocab.Collate(lang1, lang2, 4)
    )
    return loader


def main():
    loader = get_loader(config.VOCAB_DIR)
    for idx, data in enumerate(loader):
        print(data)
        break

if __name__ == '__main__':
    main()