import os 
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

from os.path import join as path_join 
from utils import (
    load_json
)

def plot_pseudo_labels(coco_obj: COCO, output_dir: str)->None:
    pass



if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)
    data_dir = path_join(parent_dir, "data")
    endo2023_dir = path_join(data_dir, "Endoscapes2023_Cutmargins")
    annots_dir = path_join(endo2023_dir, "annotations")
    sam_seg50_dir = path_join(annots_dir, "SAM_Seg50")
    train_json_path = path_join(sam_seg50_dir, "train_annotation_coco.json")    

    train_coco = COCO(annotation_file=train_json_path)