import os
import json 
from collections import defaultdict
from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU

def renumber_annotation_ids(coco):
    for new_id, ann in enumerate(coco["annotations"], start=1):
        ann["id"] = new_id
    return coco

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

def return_linear_transform(num: int, M: int = 10):
    """
    Linear transformation for SAM objects
    """

    temp = num - 1
    category_id = temp // M + 1
    return category_id

def build_file_to_imgid(coco: dict):
    # file_name -> image_id
    return {im["file_name"]: im["id"] for im in coco.get("images", [])}

def build_imgid_to_anns(coco: dict):
    # image_id -> list[ann]
    imgid2anns = defaultdict(list)
    for ann in coco.get("annotations", []):
        imgid2anns[ann["image_id"]].append(ann)
    return imgid2anns

def propagate_in_video(predictor: Sam3VideoPredictorMultiGPU, session_id: str):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame

def abs_to_rel_coords(coords:list, IMG_WIDTH:int, IMG_HEIGHT:int, coord_type:str="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")