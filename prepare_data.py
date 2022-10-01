import os
import sys
import time
import vocab
import config
import argparse
from tqdm import tqdm

def create_vocab(dataset_dir:str, use_percent:float=1, showTime=True):
    # Start performance time counter
    start = time.perf_counter()
    
    # Whether directory is existed or not
    if not os.path.exists(config.DATA_DIR):
        os.mkdir(config.DATA_DIR)
        os.mkdir(config.VOCAB_DIR)
        os.mkdir(config.TRAIN_DIR)
        os.mkdir(config.TEST_DIR)

    # If no vocabulary => create
    if len(os.listdir(config.VOCAB_DIR)) == 0:
        # open input and target file
        data = config.read_file(dataset_dir, None)
        # Initializing languages
        jp = vocab.Vocab('ja', use_jp=True)
        vi = vocab.Vocab('vi', use_vi=True)
        pbar = tqdm(data, total=len(data[0]), leave=False)
        
        # Loading sentences
        for j2v in pbar:
            jp.add_sentence(j2v[0])
            vi.add_sentence(j2v[1])
            pbar.set_description(f"Load {dataset_dir}")
        
        # Saving languages
        jp.save(config.VOCAB_DIR)
        vi.save(config.VOCAB_DIR)
    
    # End performance time counter
    end = time.perf_counter()
    if showTime:
        print(f'Finished in {round(end-start, 3)} seconds')  

def create_train_test(use_percent:float):
    pass
def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-ls','--list', 
        action='store_true', 
        help='Show available dataset'
    )
    parser.add_argument(
        '-u', '--use', 
        type=float,
        default=config.USE_PERCENT, 
        help=f'How much percentage is used (Default: {config.USE_PERCENT} ~ {config.USE_PERCENT}%%)'
    )
    parser.add_argument(
        '--create-vocab',
        type=str,
        default=config.INIT_VOCAB,
        help=f'Create Vocab (Default: {config.INIT_VOCAB})'
    )
    parser.add_argument(
        '-tr', '--train',
        type=str
    )
    args = parser.parse_args()
    return args

def run(args):
    list_dir = os.listdir(config.ORIGINAL_DS_DIR)
    if args.list:
        print('Avaible Vocabulary: ')
        [print(dir_name) for dir_name in list_dir]
        sys.exit()        
    if not os.path.exists(config.VOCAB_DIR):
        print("Creating JP-VI vocab:")
        dataset_dir = config.ORIGINAL_DS_DIR + args.create_vocab
        create_vocab(dataset_dir)
    # if args.use < 1 or args.use > 100:
    #     print(f"Invalid using percentage! Set to default {config.USE_PERCENT}")
    #     args.use = config.USE_PERCENT 
    
if __name__ == '__main__':
    args = init_args()
    run(args)