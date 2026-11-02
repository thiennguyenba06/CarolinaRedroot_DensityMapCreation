import os, numpy as np, cv2, ultralytics, re
import nms_module
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from tqdm import tqdm

# helper functions
def divideImageImproved(image_name, parent_directory, image_folder_dir, model, output_dir, img_dim, iou_thresh, conf_thresh, batchsize):
    image_path = image_folder_dir / image_name  

    base = image_path.stem
    output_name = base + ".txt"
    predicted_dir = parent_directory / output_dir
    predicted_dir.mkdir(exist_ok = True)

    img = cv2.imread(str(image_path)) # 3d array
    if img is None:
        print("Not a valid path")
        print(f"Path: {image_path}")
        return
    
    # y: vertical, x: horizontal
    y, x = img.shape[:2]

    image_f = img.copy()

    num_cols = - (x // -img_dim)
    x_overlap = img_dim - (x // num_cols)

    num_rows = - (y // -img_dim)
    y_overlap = img_dim - (y // num_rows)
    
    tiles = []
    offsets = []

    for i in range(0, x, img_dim):
        for j in range(0, y, img_dim):
            col_idx = i // img_dim
            row_idx = j // img_dim


            start_x = i - int(x_overlap * col_idx) # top left of current tile
            start_y = j - int(y_overlap * row_idx) # top left of current tile
            tile = img[start_y : start_y + img_dim, start_x : start_x + img_dim]
            tiles.append(tile)
            offsets.append((start_x, start_y)) 
    # finished creating tiles and stored offset

    # run_prediction on tiles
    results = model.predict(source = tiles, batch=batchsize, save=False, imgsz=img_dim, line_width=3, 
                            show_labels=False, show_conf=False, max_det = 3000, 
                            iou=iou_thresh, conf=conf_thresh)

    
    
    with (predicted_dir / output_name).open("w") as f: 
        detections = []
        global_conf = []

        for result, (start_x, start_y) in zip(results, offsets):
            if result.obb is not None:
                boxes = result.obb.xyxyxyxy.tolist() 
                conf_scores = result.obb.conf.tolist()
                for box, conf in zip(boxes, conf_scores):
                    # map local coordinates to global
                    global_box = [[pt[0] + start_x, pt[1] + start_y] for pt in box]
                    # normalize
                    # norm_box = [[pt[0] / x, pt[1] / y] for pt in global_box]
                    detections.append(global_box)
                    global_conf.append(conf)
        
        # perform nms
        data_abstract = np.dtype([('box', np.float32, (4, 2)), ('conf', np.float32)]) # store global box coordinates and corresponding confidence scores
        global_boxes = np.zeros(len(detections), dtype=data_abstract) # store global box coordinates and corresponding confidence scores
        for i, (box, conf) in enumerate(zip(detections, global_conf)):
            global_boxes[i] = (box, conf)
        
        nms_boxes = nms_module.nms(boxes=global_boxes, conf_threshold=conf_thresh, iou_threshold=iou_thresh)
        for box in nms_boxes:
            norm_box = [[pt[0] / x, pt[1] / y] for pt in box]
            coords_str = " ".join([f"{pt[0]} {pt[1]}" for pt in norm_box])
            f.write(f"0 {coords_str}\n")

            # draw
            pts = np.int32([box])
            cv2.polylines(image_f, pts, True, (0, 0, 255), 2)
    predicted_image_path = predicted_dir / image_path.name
    cv2.imwrite(str(predicted_image_path), image_f)


# global
def infer(parent_dir, geotagged_dir, weights_path, bog_name, sequential): 
    model = ultralytics.YOLO(str(weights_path)) 
    images_list = []
    for file in geotagged_dir.iterdir():
        if file.suffix.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.mpo')):
            images_list.append(file.name)

    output_dir = parent_dir / f"{bog_name}_predicted_frames"
    print(f"Output will be saved in: {output_dir}\n")
    output_dir.mkdir(exist_ok=True)

    if sequential: 
        for img in tqdm(images_list, desc="Doing inference on geotagged_frames"):
            divideImageImproved(image_name=img,
                                parent_directory=parent_dir,
                                image_folder_dir=geotagged_dir,
                                model=model,
                                output_dir=output_dir,
                                img_dim=640,
                                iou_thresh=0.5,
                                conf_thresh=0.35,
                                batchsize=8)
    else:
        process = partial(divideImageImproved, 
                          parent_directory=parent_dir,
                          image_folder_dir=geotagged_dir,
                          model = model, 
                          output_dir=output_dir, 
                          img_dim=640, 
                          iou_thresh=0.5,
                          conf_thresh=0.35,
                          batchsize=8) 

        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(process, images_list), total=len(images_list), desc="Doing inference on geotagged_frames"))

    print("Done!")
    return output_dir


# for testing purpose only
if __name__ == "__main__":
    # when refactored into a function take this straight from argument
    parent_directory = Path(__file__).resolve().parent
    geotagged_dir = Path("/Users/thiennguyenba/Documents/School/Research/CarolinaRedroot/Density_Video/Bog9_geotagged_frames")
    weight_path = Path("/Users/thiennguyenba/Documents/School/Research/CarolinaRedroot/Density_Video/best.pt") 
    bog_name = "Bog9" 
    sequential = True
 
    model = ultralytics.YOLO(weight_path) 
    images_list = []
    for file in geotagged_dir.iterdir():
        if file.suffix.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.mpo')):
            images_list.append(file.name)

    output_dir = parent_directory / f"{bog_name}_predicted_frames"
    print(f"Output will be saved in: {output_dir}\n")
    output_dir.mkdir(exist_ok=True)

    if sequential: 
        for img in tqdm(images_list, desc="Doing inference on geotagged_frames"):
            divideImageImproved(image_name=img,
                                parent_directory=parent_directory,
                                image_folder_dir=geotagged_dir,
                                model=model,
                                output_dir=output_dir,
                                img_dim=640,
                                iou_thresh=0.5,
                                conf_thresh=0.35,
                                batchsize=8)
    else:
        process = partial(divideImageImproved, 
                          parent_directory=parent_directory,
                          image_folder_dir=geotagged_dir,
                          model = model, 
                          output_dir=output_dir, 
                          img_dim=640, 
                          iou_thresh=0.5,
                          conf_thresh=0.35,
                          batchsize=8) 

        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(process, images_list), total=len(images_list), desc="Doing inference on geotagged_frames"))

    print("Done!")
