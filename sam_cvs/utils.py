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