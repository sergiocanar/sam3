import os
from tqdm import tqdm
from os.path import join as path_join

def load_txt(path: str) -> str:
    '''Load a text file and return its contents as a list.'''
    
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(line.strip())
        
    return data

def create_sam3_symlink(split: str, split_lt:list[int], output_dir: str) -> None:
    
    with tqdm(total=len(split_lt), desc=f"Processing {split}...", unit="video") as pbar:            
        for vid_in_split in split_lt:
            vid_name = f"video_{str(vid_in_split).zfill(3)}"
            vid_dir = path_join(frames_dir, vid_name)
            dst_dir = path_join(output_dir, vid_name)
            os.symlink(vid_dir, dst_dir)
            pbar.update(1)
                
if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = path_join(this_dir, 'data')
    endoscapes_cutmargins_dir = path_join(data_dir, 'Endoscapes2023_Cutmargins')
    frames_dir = path_join(endoscapes_cutmargins_dir, 'frames')
    annots_dir = path_join(endoscapes_cutmargins_dir, 'annotations')
    seg50_dir = path_join(annots_dir, 'Seg50')
    sam3_dir = path_join(this_dir, "sam3")
    output_dir = path_join(sam3_dir, "data", "Endoscapes2023_Cutmargins")
    os.makedirs(output_dir, exist_ok=True)
    
    
    split_lt = ['train', 'val', 'test']
    
    for split in split_lt:
        txt_info = load_txt(path=path_join(data_dir, f'{split}_seg_vids.txt'))
        txt_info = sorted([int(float(vid)) for vid in txt_info])
        split_dir = path_join(output_dir, split)
        img_dir = path_join(split_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        
        create_sam3_symlink(
            split=split,
            split_lt=txt_info,
            output_dir=img_dir
        )
