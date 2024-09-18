# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:56:20 2019

@author: gallmanj
@updated_by: mschnalke

The script was originally authored by gallmanj and has been adapted to newer models and software packages.
"""

from utils import file_utils
from utils import flower_info
from object_detection.utils import label_map_util

def non_max_suppression(detections, iou_thresh):
    """Apply non-maximum suppression to avoid overlapping bounding boxes."""
    detections = reversed(sorted(detections, key=lambda i: i['score']))
    keep = []
    
    for detection in detections:
        add_this_one = True
        for added in keep:
            if iou(added["bounding_box"], detection["bounding_box"]) >= iou_thresh:
                add_this_one = False
                break
        if add_this_one:
            keep.append(detection)
    
    return keep

def iou(a, b, epsilon=1e-5):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    width = (x2 - x1)
    height = (y2 - y1)
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    iou = area_overlap / (area_combined + epsilon)
    return iou

def get_ground_truth_annotations(image_path):
    """Read ground truth annotations from various formats."""
    ground_truth = file_utils.get_annotations(image_path)
    for fl in ground_truth:
        fl["name"] = flower_info.clean_string(fl["name"])
        fl["bounding_box"] = flower_info.get_bbox(fl)
    
    if len(ground_truth) == 0:                     
        return None
    return ground_truth

def get_index_for_flower(categories, flower_name):
    """Return the TensorFlow ID for a given flower name."""
    for flower in categories:
        if flower["name"] == flower_name:
            return flower["id"]
    raise ValueError('flower_name does not exist in categories dict')

def get_flower_names_from_labelmap(labelmap_path):
    """Convert a label map to a list of flower names and categories."""
    flower_names = []
    categories = []
    category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
    for d in category_index:
        flower_names.append(category_index[d]["name"])
        categories.append({"id": category_index[d]["id"], "name": category_index[d]["name"]})
    return (flower_names, categories)

def filter_ground_truth(ground_truths, flower_names):
    """Filter ground truth annotations by flower names."""
    return [gt for gt in ground_truths if gt["name"] in flower_names]

def filter_predictions(predictions, min_score):
    """Filter predictions by minimum score."""
    return [prediction for prediction in predictions if prediction["score"] >= min_score]
