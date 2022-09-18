import os
import time
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import concurrent.futures as cf

def google_drive_downloader(id, path, file):
    URL = "https://docs.google.com/uc?export=download"

    if not os.path.exists(path):
        os.mkdir(path)
    out_name = path + file
    temp_size = 0
    if os.path.exists(out_name):
        temp_size = int(os.path.getsize(out_name))
    
    headers = {'Range': 'bytes=%d-' % temp_size}
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True, headers=headers)
    token = get_confirm_token(response)  
    
    if token:
        params = { 'id' : id, 'confirm' : token}
        response = session.get(URL, params = params, stream = True)
    
    content_type = response.headers['content-type']
    if content_type.startswith('text/html'):
        html_content = BeautifulSoup(response.text, features='lxml')
        forms = html_content.findAll('form')
        if len(forms) == 0:
            raise ValueError("An error occured! Can't download data!")
        else: 
            response = session.get(forms[0]['action'], stream = True, headers=headers)

    save_content(response, path, file, temp_size) 

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_content(response, path, file, temp_size):
    chunk_size = 1024 * 1024 
    content_size = int(response.headers['Content-Length'])
    if content_size >= temp_size:
        total = int(content_size/chunk_size)
        initial = int(temp_size/chunk_size)
        with open(path+file, "ab") as f:
            pbar = tqdm(unit='MB', total=total, desc=file, initial=initial)
            pbar.clear()
            for chunk in response.iter_content(chunk_size=chunk_size):            
                pbar.update()
                f.write(chunk)
            pbar.close()

WINRAR_path = r'C:\"Program Files"\WinRAR\UnRAR.exe'

if __name__ == "__main__":
    path = './'
    file_ids = [
        '1pFRe9Leb_6JI7Gk-KvYPuryFh2ZfQuML', # image + text
    ]    
    file_name = [
        'data.rar',
    ]
    start_dl = time.perf_counter()
    with cf.ThreadPoolExecutor() as executor:
        results = [executor.submit(google_drive_downloader, *[file_ids[i], path, file_name[i]]) for i in range(len(file_ids))]
        i = 0
        for f in cf.as_completed(results):
            try:
                f.result()
            except:
                print('An error occured when downloading '+file_name[i]+' !')
            i += 1
    end_dl = time.perf_counter()
    print(f'Finished downloading in {end_dl-start_dl} seconds')
    
    start_unz = time.perf_counter()
    with cf.ThreadPoolExecutor() as executor:
        results = [executor.submit(os.system, WINRAR_path+' x '+path+file_name[i]+' '+path) for i in range(len(file_ids))]
        i = 0
        for f in cf.as_completed(results):
            try:
                f.result()
                os.remove(path+file_name[i])
            except:
                print('An error occured when unzipping '+file_name[i]+' !')
            i += 1
    end_unz = time.perf_counter()
    print(f'Finished unzipping in {end_unz-start_unz} seconds')