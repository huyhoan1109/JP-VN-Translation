import vocab
from torch.utils.data import DataLoader
import pickle
def get_loader(
    root,
    batch_size=1,
    shuffle=True,
    pin_memory=True
):
    j2v = [
        ['私は', 'tôi'],
        ['君', 'cậu']
    ]
    with open(f"{root}/ja.pk", "rb") as f:
        lang1 = pickle.load(f)
    with open(f"{root}/vi.pk", "rb") as f:
        lang2 = pickle.load(f)
    # print(lang1.word2index['私'])
    dataset = vocab.VocabDataset(j2v)
    #print(dataset.__getitem__(1))
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=vocab.Collate(lang1, lang2, 4)
    )
    return loader
#   print(lang1)

def main():
    loader = get_loader('vocab')
    for idx, data in enumerate(loader):
        print(data)
        break

if __name__ == '__main__':
    main()