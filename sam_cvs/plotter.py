import os 
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils 

from tqdm import tqdm
from os.path import join as path_join 
from utils import (
    load_json,
    create_dir_if_not_exists,
    build_file_to_imgid,
    build_imgid_to_anns
)

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

def plot_sequence_multimodel(
    seq_dicts_by_model: list,
    model_names: list,
    colormap: list,
    output_dir: str,
) -> None:
    """
    Plot an MxN grid:
      - rows: models (SAM2, SAM3, ...)
      - cols: time window frames (t-1, t, t+1, ...)

    seq_dicts_by_model: list of length M
        each element is a list of dicts (length N), like:
        {
            "sec": int,
            "img_id": int or None,
            "file_name": "/abs/path/to/frame.jpg",
            "annos": [coco_ann, ...] or [],
            "missing": bool
        }
    model_names: list of strings length M
    """
    create_dir_if_not_exists(output_dir)

    catid2rgb = {c["id"]: (np.array(c["color"], dtype=np.float32) / 255.0) for c in colormap}

    if len(seq_dicts_by_model) == 0:
        return

    n_models = len(seq_dicts_by_model)
    n = len(seq_dicts_by_model[0])
    if n == 0:
        return

    mid = n // 2

    fig, axs = plt.subplots(n_models, n, figsize=(5 * n, 5 * n_models))
    if n_models == 1:
        axs = np.expand_dims(axs, axis=0)
    if n == 1:
        axs = np.expand_dims(axs, axis=1)

    for r in range(n_models):
        model_seq = seq_dicts_by_model[r]
        model_name = model_names[r] if r < len(model_names) else f"model_{r}"

        for i in range(n):
            info = model_seq[i]
            ax = axs[r, i]

            img_path = info.get("file_name", None)
            missing = bool(info.get("missing", False))

            # ---- Load image ----
            if img_path is None or (not os.path.exists(img_path)):
                ax.set_axis_off()
                ax.set_title(f"{model_name}\nmissing image")
                continue

            img = plt.imread(img_path)
            ax.imshow(img)

            # ---- Overlay masks ----
            annos = info.get("annos", []) or []
            for ann in annos:
                seg = ann.get("segmentation", None)
                if seg is None:
                    continue
                try:
                    m = mask_utils.decode(seg)
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

                overlay = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.float32)
                overlay[..., 0] = rgb[0]
                overlay[..., 1] = rgb[1]
                overlay[..., 2] = rgb[2]

                ax.imshow(overlay, alpha=0.45 * m)

            # ---- Titles ----
            rel = i - mid
            time_tag = "t (KF)" if rel == 0 else f"t{rel:+d}"
            img_id = info.get("img_id", None)

            if missing:
                title = f"{model_name}\n{time_tag} | missing masks"
            else:
                title = f"{model_name}\n{time_tag} | id={img_id if img_id is not None else 'NA'}"

            ax.set_title(title, fontsize=11)
            ax.set_axis_off()

    plt.tight_layout()

    # Use center frame name for saving
    center_path = seq_dicts_by_model[0][mid].get("file_name", "sequence")
    base = os.path.splitext(os.path.basename(center_path))[0]
    parent = os.path.basename(os.path.dirname(center_path))
    out_name = f"{parent}_{base}_win{mid}_multimodel.png"
    out_path = path_join(output_dir, out_name)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_pseudo_labels_multimodel(
    frames_dir: str,
    og_coco_dict: dict,
    pseudo_coco_dicts: list,   # [sam2_dict, sam3_dict, ...]
    model_names: list,         # ["SAM2", "SAM3", ...]
    colormap: list,
    output_dir: str,
    window: int = 1,
) -> None:

    # keyframes list from GT json
    kf_imgs_lt = og_coco_dict["images"]
    kf_file_name_lt = sorted([img_info["file_name"] for img_info in kf_imgs_lt])

    # Pre-build lookup tables per model (file -> img_id, img_id -> annos)
    file2imgid_per_model = []
    imgid2annos_per_model = []
    for coco in pseudo_coco_dicts:
        file2imgid_per_model.append(build_file_to_imgid(coco=coco))
        imgid2annos_per_model.append(build_imgid_to_anns(coco=coco))

    with tqdm(total=len(kf_file_name_lt), desc="Plotting pseudo labels (multi-model)...", unit="Keyframe") as pbar:
        for kf_file in kf_file_name_lt:
            vid, frame_jpg = kf_file.split("/")
            kf_frame_num = int(frame_jpg.split(".")[0])

            min_past_frame = kf_frame_num - window
            max_future_frame = kf_frame_num + window

            seq2plot = np.arange(start=min_past_frame, stop=max_future_frame + 1, step=1)
            file2plot = [f"{vid}/{int(frame_idx):06d}.jpg" for frame_idx in seq2plot]

            # Build per-model sequences aligned by *file path* (not img_id)
            seq_dicts_by_model = []
            for m_idx in range(len(pseudo_coco_dicts)):
                file2imgid = file2imgid_per_model[m_idx]
                imgid2annos = imgid2annos_per_model[m_idx]

                model_seq = []
                for i, rel_file in enumerate(file2plot):
                    sec = int(seq2plot[i])
                    abs_path = path_join(frames_dir, rel_file)

                    if rel_file not in file2imgid:
                        # missing masks for this model on this frame
                        model_seq.append({
                            "sec": sec,
                            "img_id": None,
                            "file_name": abs_path,
                            "annos": [],
                            "missing": True
                        })
                        continue

                    img_id = file2imgid[rel_file]
                    annos = imgid2annos.get(img_id, [])

                    model_seq.append({
                        "sec": sec,
                        "img_id": img_id,
                        "file_name": abs_path,
                        "annos": annos,
                        "missing": False
                    })

                seq_dicts_by_model.append(model_seq)

            plot_sequence_multimodel(
                seq_dicts_by_model=seq_dicts_by_model,
                model_names=model_names,
                colormap=colormap,
                output_dir=output_dir
            )
            pbar.update(1)


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
            
            
def plot_image_with_segmentation(
    image_path: str,
    annos: list,
    colormap: list,
    output_path: str = None,
    alpha: float = 0.45,
) -> None:
    """
    Plot a single image with instance segmentation overlays.

    image_path: absolute path to image
    annos: list of COCO annotations (RLE in ann["segmentation"])
    colormap: list of dicts with keys: id, color [R,G,B]
    output_path: if provided, saves the image
    alpha: transparency for mask overlay
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # category_id -> RGB in [0,1]
    catid2rgb = {
        c["id"]: np.array(c["color"], dtype=np.float32) / 255.0
        for c in colormap
    }

    img = plt.imread(image_path)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)

    for ann in annos:
        seg = ann.get("segmentation", None)
        if seg is None:
            continue

        try:
            mask = mask_utils.decode(seg)
        except Exception:
            continue

        if mask.ndim == 3:
            mask = mask[..., 0]

        mask = (mask > 0).astype(np.float32)
        if mask.sum() == 0:
            continue

        cat_id = ann.get("category_id", None)
        rgb = catid2rgb.get(cat_id, np.array([1.0, 1.0, 1.0]))

        overlay = np.zeros((*mask.shape, 3), dtype=np.float32)
        overlay[..., 0] = rgb[0]
        overlay[..., 1] = rgb[1]
        overlay[..., 2] = rgb[2]

        ax.imshow(overlay, alpha=alpha * mask)

    ax.set_axis_off()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


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
    
    sam2_seg50_dir = path_join(annots_dir, "SAM2_Seg50")
    sam2_seg50_json_path =path_join(sam2_seg50_dir, "train_annotation_coco.json")
    sam2_seg50_dict = load_json(path=sam2_seg50_json_path)
        
    #Pseudo labels + Original path from SAM3
    sam3_seg50_dir = path_join(annots_dir, "SAM3_Seg201")
    sam3_seg50_json_path = path_join(sam3_seg50_dir, "train_annotation_coco.json")    
    sam3_seg50_dict = load_json(path=sam3_seg50_json_path)
        
    #Create output dir 
    output_dir = path_join(this_dir, "visualizations")
    create_dir_if_not_exists(dir_path=output_dir)
    
    file2imgid = build_file_to_imgid(coco=sam3_seg50_dict)
    imgid2annos = build_imgid_to_anns(coco=sam3_seg50_dict)
    
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

    with tqdm(total=len(list(file2imgid.keys())), desc="Plotting labels", unit="frame") as pbar:            
        for file_name in list(file2imgid.keys()):
                
            img_id = file2imgid[file_name]
            annos = imgid2annos[img_id]
            final_file_name = file_name.replace("/", "_")
        
            
            plot_image_with_segmentation(
                image_path=path_join(frames_dir, file_name),
                annos=annos,
                colormap=colormap,
                output_path=path_join(output_dir, final_file_name)
            )
            pbar.update(1)
    # plot_pseudo_labels_multimodel(
    # frames_dir=frames_dir,
    # og_coco_dict=seg50_dict,
    # pseudo_coco_dicts=[sam2_seg50_dict, sam3_seg50_dict],
    # model_names=["SAM2", "SAM3"],
    # colormap=colormap,
    # output_dir=output_dir,
    # window=1
    # )
    
    # plot_pseudo_labels(
    #     frames_dir=frames_dir, 
    #     og_coco_dict=seg50_dict,
    #     pseudo_coco_dict=sam3_seg50_dict,
    #     colormap=colormap, 
    #     output_dir=output_dir,
    #     window=1
    # )
    
    