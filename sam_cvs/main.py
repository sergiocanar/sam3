import os 
import cv2
import glob
import sam3
import torch
import logging
import argparse
import numpy as np
from PIL import Image

from tqdm import tqdm
from sam3.model_builder import build_sam3_video_model
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

from os.path import join as path_join


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d: %(message)s",
    force=True
)

def cvs_argparse():
    parser = argparse.ArgumentParser(
        description="CVS Pseudo label generation script"
    )

    parser.add_argument(
        "--is_segmentation",
        action="store_true",
        default=False,
        help="Flag to indicate if the task is segmentation (default: False for detection)",
        required=True
    )
    
    parser.add_argument(
        "--frames_type",
        type=str,
        choices=["original", "cutmargins"],
        default="cutmargins",
        help="Type of frames to process (default: cutmargins)"
    )

    return parser
def main(video_path: str, predictor):
    pass
    


if __name__ == "__main__":
    
    #Parser
    parser = cvs_argparse()
    args = parser.parse_args()
    
    #paths
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)
    data_dir = path_join(parent_dir, "data")
    
    if args.frames_type == "original":
        logging.info("Using original Endoscapes2023 frames")
        endoscapes_dir = path_join(data_dir, "Endoscapes2023")
    else:
        logging.info("Using cutmargins Endoscapes2023 frames")
        endoscapes_dir = path_join(data_dir, "Endoscapes2023_Cutmargins")
     
    #Frames dir
    frames_dir = path_join(endoscapes_dir, "frames")
    
    #Annotations dir 
    annotations_dir = path_join(endoscapes_dir, "annotations")
    
    #COCO Jsons dir 
    if args.is_segmentation:
        logging.info("Segmentation data selected. Propagating segmentation masks.")
        jsons_dir = path_join(annotations_dir, "Seg50")
    else:
        logging.info("Detection data selected. Generating segmentation masks from bboxes.")
        jsons_dir = path_join(annotations_dir, "Bbox201")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    
    sam3_model = build_sam3_video_model()
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    
    video_path_lt = os.listdir(frames_dir)
    
    
    
    # with tqdm(total=len(video_path_lt), desc)
    
    
    