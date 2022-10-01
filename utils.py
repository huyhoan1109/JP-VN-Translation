import vocab
import config
import pickle
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def get_loader(
    root:str,
    batch_size:int,
    format='.json',
    shuffle:bool=True,
    pin_memory:bool=True,
    train:float=config.USE_PERCENT/100
):
    data = config.read_file(root, format)
    with open(f"{config.VOCAB_DIR}/ja.pk", "rb") as f:
        lang1 = pickle.load(f)
    with open(f"{config.VOCAB_DIR}/vi.pk", "rb") as f:
        lang2 = pickle.load(f)
    train_data, test_data = train_test_split(data, train_size=train)
    train_set = vocab.VocabDataset(train_data)
    collate_fn = vocab.Collate(lang1, lang2)
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, # Change later
        shuffle=shuffle,
        num_workers=config.N_WORKERS, # Change later
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, # Change later
        shuffle=shuffle,
        num_workers=config.N_WORKERS, # Change later
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True
    )
    return train_loader, test_loader

def main():
    train_loader, test_loader = get_loader(
        'VN-JP-NLP-Dataset/Tatoeba_2K', 
        batch_size=32,
        format=None
    )
    for _, data in enumerate(train_loader):
        x, y = data
        print(x.shape, y.shape)
        break
if __name__ == '__main__':
    main()