import os
import json 

def load_json(path: str) -> dict:   
    '''Load a JSON file and return its contents as a dictionary.'''
     
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_json(path: str, data: dict):
    '''Save a dictionary to a JSON file.'''
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
        
def load_txt(path: str) -> str:
    '''Load a text file and return its contents as a list.'''
    
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(line.strip())
        
    return data

def create_dir_if_not_exists(dir_path: str):
    '''Create a directory if it does not exist.'''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
def xywh_to_xyxy(bbox):
    """
    Converts the format from [x,y,w,h] into [x_min, y_min, x_max, y_max].
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]