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
    xywh_to_xyxy,
    return_linear_transform
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

def sam_predictions_to_coco(masks, image_id, category_ids=None, start_ann_id=1):
    """
    Convierte predicciones de SAM en anotaciones estilo COCO.

    Args:
        masks (list[Tensor o np.ndarray]): Lista de máscaras binarias o logits (>0 se considera foreground).
        image_id (int): ID de la imagen.
        category_ids (list[int]): Lista con el category_id de cada máscara.
        start_ann_id (int): ID inicial para las anotaciones.

    Returns:
        list[dict]: lista de anotaciones estilo COCO.
        ann_id[int]: Last annotation id that was used.
    """
    annotations = []
    ann_id = start_ann_id

    for idx, mask_pred in enumerate(masks):
        # Tensor -> numpy binario
        if isinstance(mask_pred, torch.Tensor):
            mask = (mask_pred > 0).cpu().numpy().astype(np.uint8)
        else:
            mask = (mask_pred > 0).astype(np.uint8)

        # Asegurar 2D
        if mask.ndim == 3:
            mask = mask[0]

        # RLE encoding
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("utf-8")  # JSON friendly

        # Área y bbox
        area = int(mask_utils.area(rle))
        bbox = mask_utils.toBbox(rle).tolist()  # [x, y, w, h]

        # Category ID
        category_id = category_ids[idx] if category_ids is not None else 1

        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "segmentation": rle,
            "iscrowd": 0,
            "bbox": bbox,
            "area": area,
            "category_id": category_id
        })

        ann_id += 1

    return annotations, ann_id

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

def get_video_segments(video_path: str, mask_info_lt: list, predictor: Sam3TrackerPredictor, is_segmentation: bool, window: int = 1, propagation_type: str = "both")->tuple:
        
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
        
        with tqdm(total=len(kf_idx_lt), desc='Processing keyframes...') as pbar:            
            for ann_frame_idx in kf_idx_lt: 
                for out_frame_idx, out_obj_ids, low_res_masks, out_mask_logits, obj_scores in predictor.propagate_in_video(inference_state=inference_state, start_frame_idx=ann_frame_idx, max_frame_num_to_track=window, reverse=True):
                    logging.info(f'Working on keyframe: {ann_frame_idx}')
                    video_segments[out_frame_idx] = {out_obj_id: out_mask_logits[i] for i, out_obj_id in enumerate(out_obj_ids)}
                    pbar.update(1)
        
    elif propagation_type == "future":
        logging.info(f'Propagating into the future {window} frames (s).')
        
        with tqdm(total=len(kf_idx_lt), desc='Processing keyframes...') as pbar:            
            for ann_frame_idx in kf_idx_lt: 
                for out_frame_idx, out_obj_ids, low_res_masks, out_mask_logits, obj_scores in predictor.propagate_in_video(inference_state=inference_state, start_frame_idx=ann_frame_idx, max_frame_num_to_track=window, reverse=False):
                    logging.info(f'Working on keyframe: {ann_frame_idx}')
                    video_segments[out_frame_idx] = {out_obj_id: out_mask_logits[i] for i, out_obj_id in enumerate(out_obj_ids)}
                    pbar.update(1)
    else:
        raise ValueError(f'Propagation type must be: both, past or future. You have: {propagation_type}')
    
    return inference_state, video_segments, kf_idx_lt, video_kf_lt    

def update_coco_json(base_path: str, coco_json_dict: dict, video_segments: dict, kf_lt: list, img_id_counter: int, ann_id_counter: int) -> dict:
    
    #Extract lists from COCO json
    images_info_lt = coco_json_dict["images"]
    annos_info_lt = coco_json_dict["annotations"]
    cats_lt = coco_json_dict["categories"]
    video_name = base_path.split("/")[-1]

    #Frame lists from video in analysis
    frame_paths_lt = sorted(glob(path_join(base_path, "*.jpg")))
    
    
    #Itterate over frame_idx in video_segments...
    for frame_idx, obj_dict in video_segments.items():
        if frame_idx in kf_lt:
            #Skip those frame that are kf. I have their information.
            continue
        else:
            #extract frame path given the index
            frame_path = frame_paths_lt[frame_idx]
            path_json = "/".join(frame_path.split('/')[-2:])
            
            # Extract shape from img 
            img = cv2.imread(frame_path) #h, w
            h, w, _ = img.shape

            #We dont need the img anymore
            del img
            
            image_info = {
                "file_name": path_json, 
                "height": h,
                "width": w,
                "id": img_id_counter,
                "video_name": video_name,
                "frame_id": '',
                "is_det_keyframe": True,
                "ds": [],
                "video_id": int(video_name.split("_")[-1]),
                "is_ds_keyframe": False
            }
            
            images_info_lt.append(image_info)
            
            #Lists for masks and cats
            masks = []
            category_ids = []
            for obj_id, logit in obj_dict.items():
                # We need the mask as binary
                if isinstance(logit, torch.Tensor):
                    mask = (logit > 0).cpu().numpy().astype(np.uint8)
                else:
                    mask = (logit > 0).astype(np.uint8)

                if mask.ndim == 3:
                    mask = mask[0]

                if mask is not None and mask.any():
                    masks.append(mask)
                    category_ids.append(return_linear_transform(obj_id))
                    
            if len(masks) == 0:
                img_id_counter += 1
                continue
            else:
                            
                #Annots converted to COCO
                new_annos_lt, latest_ann_id = sam_predictions_to_coco(
                    masks=masks,
                    image_id=img_id_counter,
                    category_ids=category_ids,
                    start_ann_id=ann_id_counter
                )
                
                annos_info_lt.extend(new_annos_lt)
        
        img_id_counter += 1
        
    final_dict = {
        "images": images_info_lt,
        "annotations": annos_info_lt,
        "categories": cats_lt
    }
    # breakpoint()
        
    return final_dict, img_id_counter, latest_ann_id
            
            
            

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
    
    ann_id = max(img["id"] for img in gt_coco_dict["images"]) + 1
    img_id_count = max(ann["id"] for ann in gt_coco_dict["annotations"]) + 1
    final_dict = {}
    
    with tqdm(total=len(train_vids_lt), desc="Processing videos...", unit="video") as pbar:
        for video_id in train_vids_lt:
            pbar.write(f"Now processing: {video_id}")  
            video_path = path_join(frames_dir, video_id)
            inference_state, video_segments, kf_lt, filtered_mask_info_lt = get_video_segments(
                video_path=video_path,
                mask_info_lt=masks_info_lt,
                predictor=predictor,
                is_segmentation=args.is_segmentation,
            )
                          
            new_final_dict, new_img_id_counter, new_latest_ann_id = update_coco_json(
                        base_path=video_path,
                        coco_json_dict=gt_coco_dict,
                        video_segments=video_segments,
                        kf_lt=kf_lt,
                        img_id_counter=img_id_count,
                        ann_id_counter=ann_id
                    )
            
            
            final_dict = new_final_dict
            img_id_count = new_img_id_counter
            ann_id = new_latest_ann_id
        
            pbar.update(1)
    
    final_coco_json = path_join(output_dir, "train_annotation_coco.json")
    save_json(path=final_coco_json, data=final_dict)
    logging.info(f"Processing completed. Masks are saved in {output_dir}")