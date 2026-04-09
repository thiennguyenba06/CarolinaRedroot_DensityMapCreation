import os, cv2, numpy as np
from shapely import Polygon, polygons

# internal method
def iou(box1: Polygon, box2: Polygon):
    """Calculate the Intersection over Union (IoU) of two polygons."""
    return box1.intersection(box2).area / box1.union(box2).area


# global methods
def nms(boxes, conf_threshold, iou_threshold):
    """Perform Non-Maximum Suppression (NMS) on a list of bounding boxes.
    Parameters:
    boxes (numpy structured array): An array of bounding boxes with fields 'box' and 'conf'.
    conf_threshold (float): The confidence threshold to filter boxes.
    iou_threshold (float): The IoU threshold to suppress boxes.
    Returns:
    list: A list of obbs that have been filtered by NMS.
    """
    # discard box with conf < threshold
    mask = boxes['conf'] >= conf_threshold
    boxes = boxes[mask]
    # sort by conf
    boxes.sort(order='conf')  # ascending order
    boxes = boxes[::-1]  # reverse 

    # discard if iou > threshold
    final_boxes = []
    polygons = [Polygon(box) for box in boxes['box']]
    idx_list = list(range(len(polygons)))
    while idx_list:
        anchor_idx = idx_list.pop(0)
        final_boxes.append(boxes[anchor_idx]['box'])
        remaining_idx = []
        remaining_idx = [idx for idx in idx_list if iou(polygons[anchor_idx], polygons[idx]) < iou_threshold]
        idx_list = remaining_idx
    return final_boxes