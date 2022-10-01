import vocab
import config

import json
import pickle
from torch.utils.data import DataLoader

def read_file(root, format='.json'):
    data = []
    if format == '.json': 
        with open(root, 'r', encoding='utf-8') as f:
            translate = json.load(f)
        for i in translate:
            data.append([str(i), str(translate[i])])
    else:
        with open(root, 'rb', encoding='utf-8') as f:
            data = pickle.load(f)
    return data

def get_loader(
    root,
    format='.json',
    batch_size:int=1,
    shuffle:bool=True,
    pin_memory:bool=True,
):
    data = read_file(root, format)
    dataset = vocab.VocabDataset(data)
    with open(f"{config.VOCAB_DIR}/ja.pk", "rb") as f:
        lang1 = pickle.load(f)
    with open(f"{config.VOCAB_DIR}/vi.pk", "rb") as f:
        lang2 = pickle.load(f)
    dataset = vocab.VocabDataset(data)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, # Change later
        shuffle=shuffle,
        num_workers=config.N_WORKERS,
        pin_memory=pin_memory,
        collate_fn=vocab.Collate(lang1, lang2, 4)
    )
    return loader

def main():
    loader = get_loader('test.json')
    for idx, data in enumerate(loader):
        print(data)
        break

if __name__ == '__main__':
    main()