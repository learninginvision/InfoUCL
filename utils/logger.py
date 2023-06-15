import os

class Simple_Logger(object):
    '''Save test metric to file'''
    def __init__(self, save_path: str) -> None:
        if save_path.endswith('.txt'):
            self.file_path = save_path
        else:
            self.file_path = os.path.join(save_path, 'log.txt')
    
    def log(self, name, value):
        '''Add data to file'''
        with open(self.file_path, 'a+') as f:
            f.write(f"{name}: {value}\n")
            print(f"{name}: {value}")