import os 
from glob import glob
import cv2
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

from pycocotools import mask as mask_utils
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor

from os.path import join as path_join
from utils import (
    load_json,
    save_json,
    load_txt,
    create_dir_if_not_exists,
    xywh_to_xyxy
)


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
        required=False
    )
    
    parser.add_argument(
        "--frames_type",
        type=str,
        choices=["original", "cutmargins"],
        default="cutmargins",
        help="Type of frames to process (default: cutmargins)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="SAM_Seg50",
        help="Directory to save output results (default: SAM_Seg50)"
    )
    return parser

def obtain_bboxes(info: dict):
    bboxes = []
    categories = []
    for seg_info in info['segmentations_list']:
        bbox = seg_info['bbox']
        category = seg_info['category_id']
        bboxes.append(xywh_to_xyxy(bbox))
        categories.append(category)

    return bboxes, categories

def rle_list_to_masks(rle_list: list, M: int = 10)->list:
    """
    Convert a dict list with RLEs (COCO-style)
    into binary mask as PyTorch Tensor= (H, W).
    
    Args:
        rle_list (list[dict]): RLEs list (Each one in COCO format).
        M (int): Max number of instances. Default is 10 as its the maximum in Endoscapes2023

    Returns:
        masks (list[torch.Tensor]): Binary mask list (H, W).
    """

    #Lists for output
    masks = []
    categories = []
    object_ids = []
    
    #Dict for counting instances
    count_instances = {1: 1, 2: 1, 3:1, 4: 1, 5:1, 6:1}
    
    for rle in rle_list['segmentations_list']:
        mask = mask_utils.decode(rle['segmentation'])  # (H, W, 1) o (H, W)
        if mask.ndim == 3:             # a veces viene con última dim = 1
            mask = mask[..., 0]
        # mask = mask.astype(bool) #For SAM2
        
        #Masks need to be in Tensors in SAM3
        mask = torch.from_numpy(mask).bool()
        masks.append(mask)
        
        categories.append(rle['category_id'])
        object_id_instance = (rle['category_id'] - 1) * M + count_instances[rle['category_id']]
        object_ids.append(object_id_instance)
        count_instances[rle['category_id']] += 1

    return masks, object_ids, categories 

def process_json_data(coco_data: dict):
    
    '''
    Process COCO JSON data to extract image and mask information.
    Arguments:
        coco_data (dict): COCO formatted JSON data containing images and annotations.
    Returns:
        complete_info (list): List of dictionaries with image file names, IDs, video IDs, and associated segmentation masks.
    
    '''
    #Information list
    complete_info = []
    img_info_lt = coco_data['images']
    annos_info_lt = coco_data['annotations']
    
    #Iterate over images and collect masks
    for img_info in img_info_lt:
        file_name = img_info['file_name']
        img_id = img_info['id']
        video_id = img_info['video_id']
        
        img_annos_info_lt = [
            {k: v for k, v in ann.items() if k not in ['id', 'image_id']}
            for ann in annos_info_lt
            if ann['image_id'] == img_id
        ]
        
        complete_info.append({'file_name': file_name,
                              'image_id': img_id,
                              'video_id': video_id,
                               'segmentations_list': img_annos_info_lt})

    return complete_info

def main(video_path: str, mask_info_lt: list, predictor: Sam3TrackerPredictor, is_segmentation: bool, window: int = 1, propagation_type: str = "both", output_dir: str = None)->None:
        
    #Total frames lists
    frames_lt = sorted(glob(path_join(video_path, "*.jpg")))
    
    video_kf_lt = []
    #Filter information for getting ready keyframes
    for frame_path in frames_lt:
        frame_vid_id = "/".join(frame_path.split('/')[-2:])

        info = next(
            (m for m in mask_info_lt if m["file_name"] == frame_vid_id), None
            )
        
        video_kf_lt.append(info)
        

    #Start inference state
    inference_state = predictor.init_state(video_path=video_path)
    
    #Keyframe idx list
    kf_idx_lt = set()
    
    #Update the inference state given the annotations in the json
    with tqdm(total=len(video_kf_lt), desc='Adding masks to inference state...') as pbar:
        for i, kf_info in enumerate(video_kf_lt):
            if kf_info is not None:
                # logging.info(f"Processing frame: {kf_info["file_name"]}")
                initial_masks, object_ids, categories = rle_list_to_masks(kf_info)
                for j in range(len(categories)):
                    frame_idx, obj_ids, low_res_masks, video_res_masks = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=i,
                        obj_id=object_ids[j],
                        mask=initial_masks[j]
                    )  
                    kf_idx_lt.add(frame_idx)
                pbar.update(1)
            else:
                pbar.update(1)
                continue
    
    kf_idx_lt = sorted(list(kf_idx_lt))
    logging.info(f"Keyframes for {video_path}: {kf_idx_lt}")
    
    #Store video segments in dict
    video_segments = {}
    
    predictor.propagate_in_video_preflight(inference_state=inference_state)
    
    if propagation_type == "both":
        logging.info(f'Propagating into the past and future {window} frames (s).')
        with tqdm(total=len(kf_idx_lt), desc='Processing keyframes...') as pbar:            
            for ann_frame_idx in kf_idx_lt:        
                logging.info(f'Working on keyframe: {ann_frame_idx}')
                
                for out_frame_idx, out_obj_ids, low_res_masks, out_mask_logits, obj_scores in predictor.propagate_in_video(inference_state=inference_state, start_frame_idx=ann_frame_idx, max_frame_num_to_track=window, reverse=False):
                    video_segments[out_frame_idx] = {out_obj_id: out_mask_logits[i] for i, out_obj_id in enumerate(out_obj_ids)}
                
                for out_frame_idx, out_obj_ids, low_res_masks, out_mask_logits, obj_scores in predictor.propagate_in_video(inference_state=inference_state, start_frame_idx=ann_frame_idx, max_frame_num_to_track=window, reverse=True):
                    video_segments[out_frame_idx] = {out_obj_id: out_mask_logits[i] for i, out_obj_id in enumerate(out_obj_ids)}
                
                pbar.update(1)
            
    elif propagation_type == "past":
        logging.info(f'Propagating into the past {window} frames (s).')
        
        logging.info(f'Working on keyframe: {ann_frame_idx}')
        for out_frame_idx, out_obj_ids, low_res_masks, out_mask_logits, obj_scores in predictor.propagate_in_video(inference_state=inference_state, start_frame_idx=ann_frame_idx, max_frame_num_to_track=window, reverse=True):
            video_segments[out_frame_idx] = {out_obj_id: out_mask_logits[i] for i, out_obj_id in enumerate(out_obj_ids)}
        
    elif propagation_type == "future":
        logging.info(f'Propagating into the future {window} frames (s).')
        
        logging.info(f'Working on keyframe: {ann_frame_idx}')
        for ann_frame_idx in kf_idx_lt:        
            for out_frame_idx, out_obj_ids, low_res_masks, out_mask_logits, obj_scores in predictor.propagate_in_video(inference_state=inference_state, start_frame_idx=ann_frame_idx, max_frame_num_to_track=window, reverse=False):
                video_segments[out_frame_idx] = {out_obj_id: out_mask_logits[i] for i, out_obj_id in enumerate(out_obj_ids)}
    else:
        raise ValueError(f'Propagation type must be: both, past or future. You have: {propagation_type}')
    
    
    
    
    breakpoint()
    
        

    


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

    #SAM initialization
    
    sam3_model = build_sam3_video_model()
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    
    #Video list
    if args.is_segmentation:
        train_vids_file = path_join(data_dir, "train_seg_vids.txt")
    else:
        train_vids_file = path_join(data_dir, "train_vids.txt")
    
    #Load train vids list    
    train_vids_lt = load_txt(train_vids_file)
    train_vids_lt = sorted([f"video_{int(float(vid_id)):03d}" for vid_id in train_vids_lt])
    
    #Lood COCO json files
    gt_coco_path = path_join(jsons_dir, "train_annotation_coco.json")
    gt_coco_dict = load_json(gt_coco_path)
    
    #Get relevant information for mask extension 
    masks_info_lt = process_json_data(coco_data=gt_coco_dict)
    
    #Output dir 
    output_dir = path_join(annotations_dir, args.output_dir)
    create_dir_if_not_exists(output_dir)
    
    ann_id = 1
    img_id_count = 1
    
    with tqdm(total=len(train_vids_lt), desc="Processing videos...", unit="video") as pbar:
        for video_id in train_vids_lt:
            pbar.write(f"Now processing: {video_id}")  
            video_path = path_join(frames_dir, video_id)
            main(
                video_path=video_path,
                mask_info_lt=masks_info_lt,
                predictor=predictor,
                is_segmentation=args.is_segmentation,
                output_dir=output_dir
            )
            pbar.update(1)
    logging.info(f"Processing completed. Masks are saved in {output_dir}")