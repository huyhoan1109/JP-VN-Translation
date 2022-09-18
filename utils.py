import os
import time
import vocab
import config
import argparse
from tqdm import tqdm

def _getlines_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    f.close()
    return lines

def preprocess(dataset_dir, use:float=1, j2v:bool=True):
    # Start performance time counter
    start = time.perf_counter()
    
    # Whether directory is existed or not
    if not os.path.exists(config.VOCAB_DIR):
        os.mkdir(config.VOCAB_DIR)
    
    # If no vocabulary => create
    if len(os.listdir(config.VOCAB_DIR)) == 0:
        # Creating in and target tokens
        input_file = dataset_dir + '/data-ja.txt' if j2v else dataset_dir + '/data-vi.txt'
        output_file = dataset_dir + '/data-vi.txt' if j2v else dataset_dir + '/data-ja.txt'
        
        # open input and target file
        input_lines = _getlines_txt(input_file)
        out_lines = _getlines_txt(output_file)
        use_len = int(len(input_lines) * use)

        use_input_lines = input_lines[:use_len]
        use_out_lines = out_lines[:use_len]

        # Initializing languages
        input_lang = vocab.Vocab('ja', use_jp=True) if j2v else vocab.Vocab('vi', use_vi=True)
        out_lang = vocab.Vocab('vi', use_vi=True) if j2v else vocab.Vocab('ja', use_jp=True)
        pbar = tqdm(zip(use_input_lines, use_out_lines), total=use_len, leave=False)
        
        # Loading sentences
        for input_sen, output_sen in pbar:
            input_lang.add_sentence(input_sen)
            out_lang.add_sentence(output_sen)
            pbar.set_description(f"Load {dataset_dir}")
        
        # Saving languages
        input_lang.save(config.VOCAB_DIR)
        out_lang.save(config.VOCAB_DIR)
    
    # End performance time counter
    end = time.perf_counter()
    print(f'Finished processing in {round(end-start, 3)} seconds')

def init_args(args):
    if args.list:
        [print(file) for file in os.listdir(config.DATASETS_DIR)]
    if args.choose:
        chosen_dir = config.DATASETS_DIR+args.choose
        if not os.path.exists(chosen_dir):
            raise ValueError("Can't find the choosen dataset")
        preprocess(chosen_dir, args.use, args.j2v)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ls','--list', action='store_true', help='Show available dataset')
    parser.add_argument('-c', '--choose', type=str, help='Choosing available dataset')
    parser.add_argument('-u', '--use', type=float, default=0.5, help='How much dataset used to create vocab')
    parser.add_argument('-j2v', '--j2v', choices=[True, False], default=True, help='How much dataset used to create vocab')
    args = parser.parse_args()
    init_args(args)
    