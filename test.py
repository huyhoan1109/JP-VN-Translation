from janome.dic import UserDictionary
from janome import sysdic
from janome.progress import SimpleProgressIndicator
path = './VN-JP-NLP-Dataset/Tatoeba_2K/'
user_dict = UserDictionary(path+'data.txt', "utf8", "simpledic", sysdic.connections, progress_handler=SimpleProgressIndicator(update_frequency=0.01))
user_dict.save("./data")