# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:50:29 2019

@author: johan

@updated_by: mschnalke

The script was originally authored by gallmanj and has been adapted to newer models and software packages.

This script makes predictions on images of any size.
"""

print("Loading libraries...")
from utils import constants
import os
from utils import file_utils
from utils import flower_info
import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils
import progressbar
from osgeo import gdal
from utils import eval_utils
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
import time

def predict(project_dir, images_to_predict, output_folder, tile_size, prediction_overlap, min_confidence_score=constants.min_confidence_score, visualize_predictions=True, visualize_groundtruths=False, visualize_scores=False, visualize_names=False, max_iou=constants.max_iou):
    """
    Makes predictions on all images in the images_to_predict folder and saves them to the
    output_folder with the prediction bounding boxes drawn onto the images.
    """
    MODEL_DIR = os.path.join(project_dir, "trained_inference_graphs", "saved_model")
    PATH_TO_LABELS = os.path.join(project_dir, "model_inputs", "label_map.pbtxt")

    # Load the model
    detect_fn = tf.saved_model.load(MODEL_DIR)

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    print(category_index)

    all_images = file_utils.get_all_images_in_folder(images_to_predict)
    
    total_inference_time = 0.0  # Variable to store the total inference time
    num_images = len(all_images)  # Total number of images
    
    for image_path in all_images:
        large_image = False
        try:
            image = Image.open(image_path)
            width, height = image.size
        except Image.DecompressionBombError:
            ds = gdal.Open(image_path)
            band = ds.GetRasterBand(1)
            width = band.XSize
            height = band.YSize
            image_array = ds.ReadAsArray().astype(np.uint8)
            image_array = np.swapaxes(image_array, 0, 1)
            image_array = np.swapaxes(image_array, 1, 2)
            large_image = True
            print("Please note that the image is very large. The prediction algorithm will work as usual but no information will be drawn onto the bounding boxes.")

        print("Making Predictions for " + os.path.basename(image_path) + "...")

        start_time = time.time()  # Start time for inference
        
        detections = []
        for x_start in progressbar.progressbar(range(-prediction_overlap, width - 1, tile_size - 2 * prediction_overlap)):
            for y_start in range(-prediction_overlap, height - 1, tile_size - 2 * prediction_overlap):
                if not large_image:
                    crop_rectangle = (x_start, y_start, x_start + tile_size, y_start + tile_size)
                    cropped_im = image.crop(crop_rectangle)
                else:
                    pad_front_x = max(0, -x_start)
                    pad_front_y = max(0, -y_start)
                    cropped_array = image_array[y_start + pad_front_y:y_start + tile_size, x_start + pad_front_x:x_start + tile_size, :]
                    pad_end_x = tile_size - cropped_array.shape[1] - pad_front_x
                    pad_end_y = tile_size - cropped_array.shape[0] - pad_front_y
                    cropped_array = np.pad(cropped_array, ((pad_front_y, pad_end_y), (pad_front_x, pad_end_x), (0, 0)), mode='constant', constant_values=0)
                    cropped_im = Image.fromarray(cropped_array)
                
                cropped_im = cropped_im.convert("RGB")
                extrema = cropped_im.convert("L").getextrema()
                if extrema[0] == extrema[1]:
                    continue
                image_np = np.asarray(cropped_im)         
                image_expand = np.expand_dims(image_np, 0)

                # Run inference
                output_dict = detect_fn(image_expand)

                # Process the output
                num_detections = int(output_dict.pop('num_detections'))
                output_dict = {key: value[0, :num_detections].numpy()
                               for key, value in output_dict.items()}
                output_dict['num_detections'] = num_detections
                output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

                core_overlap = int(prediction_overlap * 0.2)
                for i, score in enumerate(output_dict['detection_scores']):
                    center_x = (output_dict['detection_boxes'][i][3] + output_dict['detection_boxes'][i][1]) / 2 * tile_size
                    center_y = (output_dict['detection_boxes'][i][2] + output_dict['detection_boxes'][i][0]) / 2 * tile_size
                    if score >= min_confidence_score and center_x >= prediction_overlap - core_overlap and center_y >= prediction_overlap - core_overlap and center_x < tile_size - prediction_overlap + core_overlap and center_y < tile_size - prediction_overlap + core_overlap:
                        top = round(output_dict['detection_boxes'][i][0] * tile_size + y_start)
                        left = round(output_dict['detection_boxes'][i][1] * tile_size + x_start)
                        bottom = round(output_dict['detection_boxes'][i][2] * tile_size + y_start)
                        right = round(output_dict['detection_boxes'][i][3] * tile_size + x_start)
                        detection_class = output_dict['detection_classes'][i]
                        detections.append({"bounding_box": [top, left, bottom, right], "score": float(score), "name": category_index[detection_class]["name"]})
                        
        end_time = time.time()  # End time for inference
        inference_time = end_time - start_time
        total_inference_time += inference_time  # Add to the total inference time
        print(f"Inference time for {os.path.basename(image_path)}: {inference_time:.2f} seconds")
        
        detections = eval_utils.non_max_suppression(detections, max_iou)
        
        print(str(len(detections)) + " objects detected")
        predictions_out_path = os.path.join(output_folder, os.path.basename(image_path)[:-4] + ".xml")
        file_utils.save_annotations_to_xml(detections, image_path, predictions_out_path)
        ground_truth = get_ground_truth_annotations(image_path)
        if ground_truth:
            if visualize_groundtruths:
                for detection in ground_truth:
                    [top, left, bottom, right] = detection["bounding_box"]
                    col = "black"
                    if not large_image:
                        visualization_utils.draw_bounding_box_on_image(image, top, left, bottom, right, display_str_list=(), thickness=1, color=col, use_normalized_coordinates=False)          
                    else:
                        draw_bounding_box_onto_array(image_array, top, left, bottom, right)

            ground_truth_out_path = os.path.join(output_folder, os.path.basename(image_path)[:-4] + "_ground_truth.xml")
            file_utils.save_annotations_to_xml(ground_truth, image_path, ground_truth_out_path)

        for detection in detections:
            if visualize_predictions:
                col = flower_info.get_color_for_flower(detection["name"])
                [top, left, bottom, right] = detection["bounding_box"]
                score_string = str('{0:.2f}'.format(detection["score"]))
                if not large_image:
                    vis_string_list = []
                    if visualize_scores:
                        vis_string_list.append(score_string)
                    if visualize_names:
                        vis_string_list.append(detection["name"])                            
                    visualization_utils.draw_bounding_box_on_image(image, top, left, bottom, right, display_str_list=vis_string_list, thickness=1, color=col, use_normalized_coordinates=False)          
                else:
                    col = flower_info.get_color_for_flower(detection["name"], get_rgb_value=True)[0:3]
                    draw_bounding_box_onto_array(image_array, top, left, bottom, right, color=col)
        
        if visualize_groundtruths or visualize_predictions:
            image_output_path = os.path.join(output_folder, os.path.basename(image_path))
            if not large_image:
                image.save(image_output_path)
            else:
                print("Saving image. This might take a while...")
                file_utils.save_array_as_image(image_output_path[:-4] + ".png", image_array, tile_size=5000)

    # Calculate and print overall inference time and average per image
    average_inference_time = total_inference_time / num_images
    print(f"Total inference time for all images: {total_inference_time:.2f} seconds")
    print(f"Average inference time per image: {average_inference_time:.2f} seconds")

def draw_bounding_box_onto_array(array, top, left, bottom, right, color=[0, 0, 0]):
    """Draw bounding box on image array."""
    try:
        color = np.array(color).astype(np.uint8)
        top = max(0, int(top))
        left = max(0, int(left))
        bottom = min(array.shape[0] - 1, int(bottom))
        right = min(array.shape[1] - 1, int(right))
        for i in range(top, bottom + 1, 1):
            array[i, left] = color
            array[i, right] = color
        for i in range(left, right + 1):
            array[top, i] = color
            array[bottom, i] = color

    except IndexError:
        print("Index error: " + str((top, left, bottom, right)))

def get_ground_truth_annotations(image_path):
    """Reads the ground truth information from annotations."""
    ground_truth = file_utils.get_annotations(image_path)
    for fl in ground_truth:
        fl["name"] = flower_info.clean_string(fl["name"])
        fl["bounding_box"] = flower_info.get_bbox(fl)
    
    if len(ground_truth) == 0:                     
        return None
    return ground_truth

if __name__ == '__main__':
    train_dir = constants.train_dir
    images_to_predict = constants.images_to_predict
    output_folder = constants.predictions_folder
    tile_size = constants.tile_size
    prediction_overlap = constants.prediction_overlap

    predict(train_dir, images_to_predict, output_folder, tile_size, prediction_overlap)
