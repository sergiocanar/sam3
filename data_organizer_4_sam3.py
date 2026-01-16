import os

def load_txt(path: str) -> str:
    '''Load a text file and return its contents as a list.'''
    
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(line.strip())
        
    return data
