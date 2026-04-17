import os, numpy as np, cv2, ultralytics, re
import nms_module
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def divideImageImproved(image_name, parent_directory, image_folder_dir, weight_path, output_dir, img_dim, iou_thresh, conf_thresh, batchsize):
    image_path = os.path.join(parent_directory, image_folder_dir, image_name)
    model = ultralytics.YOLO(os.path.join(parent_directory, weight_path))
    base = os.path.basename(image_path)
    output_name = str(base).split(".")[0] + ".txt"

    img = cv2.imread(image_path) # 3d array
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

    with open(os.path.join(parent_directory, output_dir, output_name), "w") as f: 
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
    cv2.imwrite(os.path.join(parent_directory, output_dir, base), image_f)


if __name__ == "__main__":
    # parent_directory = (ans + "/" if (ans := input("Enter full path to parent directory: ").strip()) != "-1" else "./")
    # image_folder_dir = input("Enter relative path to image folder: ").strip() + "/"
    # weight_path = (ans if (ans := input("Enter relative path to weight file: ").strip()) != "-1" else "best.pt") 

    # for timing purposes
    parent_directory = "./"
    image_folder_dir = "geotagged_frames2"
    weight_path = "best.pt"
    
    # model = ultralytics.YOLO(os.path.join(parent_directory, weight_path))
    images = os.listdir(os.path.join(parent_directory, image_folder_dir))
    images_list = []
    
    for file in images:
       if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.mpo')):
           images_list.append(file)
    
    
    # Logic to handle output_dir
    # use regex to check if output dir exists, if exists, find max number and add 1 then create output{num} 
    # if not, create output dir
    pattern = re.compile(r"^output(\d*)$")
    counter = 1
    for directory in os.listdir(parent_directory):
       if os.path.isdir(os.path.join(parent_directory, directory)):
           match = pattern.match(directory)
           if match:
               counter = max(counter, int(match.group(1) if match.group(1) != '' else 0)) + 1
               
    output_dir = f"output{counter}/" if counter > 1 else "output/"
    print(f"Output will be saved in: {output_dir}\n")
    os.mkdir(os.path.join(parent_directory, output_dir))

    process = partial(divideImageImproved, 
                      parent_directory=parent_directory,
                      image_folder_dir=image_folder_dir,
                      weight_path=weight_path, 
                      output_dir=output_dir, 
                      img_dim=640, 
                      iou_thresh=0.5,
                      conf_thresh=0.35,
                      batchsize=8)

    print("executing...")
    # for img in images_list:
    #     divideImageImproved(image_name=img,
    #                         parent_directory=parent_directory,
    #                         image_folder_dir=image_folder_dir,
    #                         weight_path=weight_path,
    #                         output_dir=output_dir,
    #                         img_dim=640,
    #                         iou_thresh=0.5,
    #                         conf_thresh=0.35,
    #                         batchsize=8)
    

    with ProcessPoolExecutor() as executor:
        executor.map(process, images_list) 

    print("Done!")