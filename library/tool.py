import sys
import torch
import toml

# GPUが使用可能か判断
def is_cuda_available():
    if torch.cuda.is_available():
        print('gpu is available')
    else:
        raise Exception('gpu is NOT available')

# Goole Colab環境か判断
def is_colabo():
    moduleList = sys.modules
    if 'google.colab' in moduleList:
        return True
    else:
        return False

# model, tokenizerの一覧読み取り
class ReadModelTokenizerTome():

    def __init__(self, path):
        self.path = path
        self.load(self.path)

    def __str__(self):
        return self.get_str()

    def __repr__(self) -> str:
        return toml.dumps(self.settings)

    def get_str(self):
        msg = f'tokenizer:{self.__tokenizer}\n' + \
              f'model:{self.__model}\n' + \
              f'score_folder:{self.__score_folder}'
        return msg

    def load(self, path):
        with open(self.path) as f:
            self.settings = toml.load(f)
        self.__tokenizer = ''
        self.__model = ''
        self.__score_folder = ''

    def read(self, pattern):
        self.__tokenizer = self.settings[pattern]['tokenizer']
        self.__model = self.settings[pattern]['model']
        self.__score_folder = self.settings[pattern]['score_folder']
    
    @property
    def tokenizer(self):
        return self.__tokenizer

    @property
    def model(self):
        return self.__model
    
    @property
    def score_folder(self):
        return self.__score_folder