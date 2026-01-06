import os 
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils 

from tqdm import tqdm
from collections import defaultdict
from os.path import join as path_join 
from utils import (
    load_json,
    create_dir_if_not_exists
)

def build_file_to_imgid(coco: dict):
    # file_name -> image_id
    return {im["file_name"]: im["id"] for im in coco.get("images", [])}

def build_imgid_to_anns(coco: dict):
    # image_id -> list[ann]
    imgid2anns = defaultdict(list)
    for ann in coco.get("annotations", []):
        imgid2anns[ann["image_id"]].append(ann)
    return imgid2anns


def plot_sequence(seq_dict_lt: list, colormap: list, output_dir: str) -> None:
    """
    Plot a 1xN sequence of frames with instance masks overlaid.

    seq_dict_lt: list of dicts like:
        {
            "sec": int,
            "img_id": int,
            "file_name": "/abs/path/to/frames/video_xxx/000123.jpg",
            "annos": [coco_ann, ...]   # coco_ann["segmentation"] is RLE
        }
    colormap: list of dicts with keys: id, name, color [R,G,B]
    output_dir: where to save the figure
    """
    create_dir_if_not_exists(output_dir)

    # category_id -> (r,g,b) in [0,1]
    catid2rgb = {c["id"]: (np.array(c["color"], dtype=np.float32) / 255.0) for c in colormap}

    n = len(seq_dict_lt)
    if n == 0:
        return

    mid = n // 2  # center frame
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axs = [axs]

    for i, info in enumerate(seq_dict_lt):
        ax = axs[i]
        img_path = info.get("file_name", None)

        # --- Load image ---
        if img_path is None or (not os.path.exists(img_path)):
            ax.set_axis_off()
            ax.set_title("missing")
            continue

        img = plt.imread(img_path)
        ax.imshow(img)

        # --- Overlay masks ---
        annos = info.get("annos", []) or []
        for ann in annos:
            seg = ann.get("segmentation", None)
            if seg is None:
                continue

            try:
                m = mask_utils.decode(seg)  # (H,W) or (H,W,1)
            except Exception:
                continue

            if m is None:
                continue
            if m.ndim == 3:
                m = m[..., 0]
            m = (m > 0).astype(np.float32)

            if m.sum() == 0:
                continue

            cat_id = ann.get("category_id", None)
            rgb = catid2rgb.get(cat_id, np.array([1.0, 1.0, 1.0], dtype=np.float32))

            # show per-instance overlay (alpha masked)
            overlay = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.float32)
            overlay[..., 0] = rgb[0]
            overlay[..., 1] = rgb[1]
            overlay[..., 2] = rgb[2]

            ax.imshow(overlay, alpha=0.45 * m)

        # --- Titles (t-1, t, t+1) ---
        rel = i - mid
        if rel == 0:
            title = f"t (KF) | id={info.get('img_id', 'NA')}"
        else:
            title = f"t{rel:+d} | id={info.get('img_id', 'NA')}"
        ax.set_title(title, fontsize=12)
        ax.set_axis_off()

    plt.tight_layout()

    # --- Save figure (use center frame name) ---
    center_path = seq_dict_lt[mid].get("file_name", "sequence")
    base = os.path.splitext(os.path.basename(center_path))[0]
    parent = os.path.basename(os.path.dirname(center_path))  # e.g., video_004
    out_name = f"{parent}_{base}_win{mid}.png"
    out_path = path_join(output_dir, out_name)  
    
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def plot_pseudo_labels(frames_dir: str, og_coco_dict:dict, pseudo_coco_dict:dict, colormap:list, output_dir: str, window: int=1)->None:
    
    kf_imgs_lt = og_coco_dict["images"]
    kf_file_name_lt = sorted([img_info["file_name"] for img_info in kf_imgs_lt])
    file2imgid_dict = build_file_to_imgid(coco=pseudo_coco_dict)
    img_id_annos = build_imgid_to_anns(coco=pseudo_coco_dict)
    
    with tqdm(total=len(kf_file_name_lt), desc="Plotting pseudo labels...", unit="Keyframe") as pbar:
            
        for kf_file in kf_file_name_lt:
            
            vid, frame_jpg = kf_file.split("/")
            kf_frame_num = frame_jpg.split(".")[0]
            kf_frame_num = int(kf_frame_num)
            min_past_frame = kf_frame_num - window
            max_future_frame = kf_frame_num + window
            
            seq2plot = np.arange(start=min_past_frame,
                                stop=max_future_frame+1,
                                step=1)
            
            file2plot = [
                f"{vid}/{int(frame_idx):06d}.jpg"
                for frame_idx in seq2plot
            ]
            

            seq2plot_dict_lt = []
            skip=False
            for i, file_path in enumerate(file2plot):
                
                sec = seq2plot[i]
                
                if file_path not in list(file2imgid_dict.keys()):
                    print(f"Skipping frame: {file_path}. Not segmentation created.")
                    skip = True
                    continue
                    
                
                img_id = file2imgid_dict[file_path]
                img_annos = img_id_annos[img_id]
                
                img2plot_dict = {
                    "sec": sec,
                    "img_id": img_id, 
                    "file_name": path_join(frames_dir, file_path),
                    "annos": img_annos
                }
                seq2plot_dict_lt.append(img2plot_dict)
            
            if skip:
                pbar.update(1)
                continue
            else:    
                plot_sequence(seq_dict_lt=seq2plot_dict_lt,
                            colormap=colormap,
                            output_dir=output_dir
                        )
                
            pbar.update(1)            
            

if __name__ == "__main__":
    
    #Essential paths 
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)
    data_dir = path_join(parent_dir, "data")
    endo2023_dir = path_join(data_dir, "Endoscapes2023")
    frames_dir = path_join(endo2023_dir, "frames")
    annots_dir = path_join(endo2023_dir, "annotations")
    
    #Original Seg50 path
    seg_50_dir = path_join(annots_dir, "Seg50")
    seg50_json_path = path_join(seg_50_dir, "train_annotation_coco.json")    
    seg50_dict = load_json(path=seg50_json_path)
    
    #Pseudo labels + Original path
    sam_seg50_dir = path_join(annots_dir, "SAM_Seg50")
    sam_seg50_json_path = path_join(sam_seg50_dir, "train_annotation_coco.json")    
    sam_seg50_dict = load_json(path=sam_seg50_json_path)
        
    #Create output dir 
    output_dir = path_join(this_dir, "visualizations")
    create_dir_if_not_exists(dir_path=output_dir)
    
    
    colormap =  [
        {
            "id": 1,
            "name": "cystic_plate",
            "supercategory": "anatomy",
            "color": [
                248,
                231,
                28
            ]
        },
        {
            "id": 2,
            "name": "calot_triangle",
            "supercategory": "anatomy",
            "color": [
                74,
                144,
                226
            ]
        },
        {
            "id": 3,
            "name": "cystic_artery",
            "supercategory": "anatomy",
            "color": [
                218,
                13,
                15
            ]
        },
        {
            "id": 4,
            "name": "cystic_duct",
            "supercategory": "anatomy",
            "color": [
                65,
                117,
                6
            ]
        },
        {
            "id": 5,
            "name": "gallbladder",
            "supercategory": "anatomy",
            "color": [
                126,
                211,
                33
            ]
        },
        {
            "id": 6,
            "name": "tool",
            "supercategory": "tool",
            "color": [
                245,
                166,
                35
            ]
        }
    ]

    
    plot_pseudo_labels(
        frames_dir=frames_dir, 
        og_coco_dict=seg50_dict,
        pseudo_coco_dict=sam_seg50_dict,
        colormap=colormap, 
        output_dir=output_dir,
        window=1
    )
    
    