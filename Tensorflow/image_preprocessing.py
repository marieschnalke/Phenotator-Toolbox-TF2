# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:25:16 2019

@author: johan
@updated_by: mschnalke

The script was originally authored by gallmanj and has been adapted to newer models and software packages.


Running this script, namely its main function convert_annotation_folders, converts one or multiple input folders containing annotated
images into a format that is readable for the Tensorflow library. The input folders
must contain images (jpg, png or tif) and along with each image a json file containing the
annotations. These json files can either be created with the widely used LabelMe Application
or with the AnnotationApp available for Android Tablets.

Output of this string is a folder structure that is used by all subsequent scripts such as
train, predict, evaluate or export-inference-graph scripts and more.
"""



print("Loading libraries...")
from utils import constants
import os
import xml.etree.ElementTree as ET
from PIL import Image
from shutil import move, copy
import utils.xml_to_csv as xml_to_csv
import random
import utils.generate_tfrecord as generate_tfrecord
from utils import flower_info, file_utils
import progressbar
import sys
import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
import urllib.request
import tarfile
import shutil
from object_detection.protos import preprocessor_pb2
import numpy as np
import colorsys


def convert_annotation_folders(input_folders, test_splits, validation_splits, project_dir, tile_sizes, split_mode, min_flowers, overlap, model_link, tensorflow_tile_size):
    """Converts the contents of a list of input folders into tensorflow readable format ready for training"""
    if len(input_folders) > len(test_splits) or len(input_folders) > len(validation_splits):
        print("Error: Make sure that you provide the same number of input folders and split values")
        return

    make_training_dir_folder_structure(project_dir)
    download_pretrained_model(project_dir, model_link)
    labels = {}
    train_images_dir = os.path.join(project_dir, "images", "train")
    test_images_dir = os.path.join(project_dir, "images", "test")
    test_images_dir_full_size = os.path.join(project_dir, "images", "test_full_size")
    validation_images_dir = os.path.join(project_dir, "images", "validation")
    validation_images_dir_full_size = os.path.join(project_dir, "images", "validation_full_size")

    for input_folder_index in range(len(input_folders)):
        input_folder = input_folders[input_folder_index]
        image_paths = file_utils.get_all_images_in_folder(input_folder)
        print(f"\nTiling images (and annotations) into chunks suitable for TensorFlow training: ({input_folder})")
        sys.stdout.flush()
        for i in progressbar.progressbar(range(len(image_paths))):
            image_path = image_paths[i]
            tile_image_and_annotations(image_path, train_images_dir, labels, input_folder_index, tile_sizes, overlap)

    flowers_to_use = filter_labels(labels, min_flowers)
    print("Creating Labelmap file...")
    annotations_dir = os.path.join(project_dir, "model_inputs")
    write_labels_to_labelmapfile(flowers_to_use, annotations_dir)

    print("Splitting train dir into train, test and validation dir...")
    labels_test = {}
    split_train_dir(train_images_dir, test_images_dir, labels, labels_test, split_mode, input_folders, test_splits, full_size_splitted_dir=test_images_dir_full_size)
    labels_validation = {}
    validation_splits = [v / (1 - min(0.9999, t)) for v, t in zip(validation_splits, test_splits)]
    split_train_dir(train_images_dir, validation_images_dir, labels, labels_validation, split_mode, input_folders, validation_splits, full_size_splitted_dir=validation_images_dir_full_size)

    # Ensure minimum class presence in training data
    ensure_minimum_class_presence(train_images_dir, test_images_dir, validation_images_dir, labels, labels_test, labels_validation, flowers_to_use)

    print("Converting Annotation data into tfrecord files...")
    train_csv = os.path.join(annotations_dir, "train_labels.csv")
    test_csv = os.path.join(annotations_dir, "test_labels.csv")
    xml_to_csv.xml_to_csv(train_images_dir, train_csv, flowers_to_use=flowers_to_use)
    xml_to_csv.xml_to_csv(test_images_dir, test_csv, flowers_to_use=flowers_to_use)
    train_tf_record = os.path.join(annotations_dir, "train.record")
    generate_tfrecord.make_tfrecords(train_csv, train_tf_record, train_images_dir, labels)
    test_tf_record = os.path.join(annotations_dir, "test.record")
    generate_tfrecord.make_tfrecords(test_csv, test_tf_record, test_images_dir, labels)
    set_config_file_parameters(project_dir, len(flowers_to_use), tensorflow_tile_size)

    print("tfrecord training files generated from the following amount of flowers:")
    print_labels(labels, flowers_to_use)
    print("the test data contains the following amount of flowers:")
    print_labels(labels_test, flowers_to_use)
    print("the validation data contains the following amount of flowers:")
    print_labels(labels_validation, flowers_to_use)
    print(f"{len(flowers_to_use)} classes usable for training.")
    print("Done!")


def ensure_minimum_class_presence(train_dir, test_dir, validation_dir, labels, labels_test, labels_validation, flowers_to_use):
    """Ensures that all classes have at least one instance in the training data."""
    images_test = file_utils.get_all_images_in_folder(test_dir)
    images_validation = file_utils.get_all_images_in_folder(validation_dir)

    for flower in flowers_to_use:
        if labels.get(flower, 0) == 0:
            # Try to find the class in test data
            found = False
            for image_path in images_test:
                if flower in file_utils.get_annotations(image_path):
                    move_image_and_annotations_to_folder(image_path, train_dir, labels_test, labels)
                    found = True
                    break
            # If not found in test, try validation data
            if not found:
                for image_path in images_validation:
                    if flower in file_utils.get_annotations(image_path):
                        move_image_and_annotations_to_folder(image_path, train_dir, labels_validation, labels)
                        break


def tile_image_and_annotations(image_path, output_folder, labels, input_folder_index, tile_sizes, overlap):
    """Tiles the image and the annotations into square shaped tiles of size tile_size"""
    image_array = file_utils.get_image_array(image_path)
    height, width = image_array.shape[:2]
    image_name = os.path.basename(image_path)
    for tile_size in tile_sizes:
        currentx, currenty = 0, 0
        while currenty < height:
            while currentx < width:
                filtered_annotations = get_flowers_within_bounds(image_path, currentx, currenty, tile_size)
                if not filtered_annotations:
                    currentx += tile_size - overlap
                    continue
                cropped_array = image_array[currenty:currenty+tile_size, currentx:currentx+tile_size, :]
                pad_end_x, pad_end_y = tile_size - cropped_array.shape[1], tile_size - cropped_array.shape[0]
                cropped_array = np.pad(cropped_array, ((0, pad_end_y), (0, pad_end_x), (0, 0)), mode='constant', constant_values=0)
                tile = Image.fromarray(cropped_array)
                output_image_path = os.path.join(output_folder, f"{image_name}_subtile_x{currentx}y{currenty}size{tile_size}_inputdir{input_folder_index}.png")
                tile.save(output_image_path, "PNG")
                xml_path = output_image_path[:-4] + ".xml"
                annotations_xml = build_xml_tree(filtered_annotations, output_image_path, labels)
                annotations_xml.write(xml_path)
                currentx += tile_size - overlap
            currentx = 0
            currenty += tile_size - overlap


def rgb2hsv(img):
    if isinstance(img, Image.Image):
        r, g, b = img.split()
        Hdat = []
        Sdat = []
        Vdat = []
        for rd, gn, bl in zip(r.getdata(), g.getdata(), b.getdata()):
            h, s, v = colorsys.rgb_to_hsv(rd / 255., gn / 255., bl / 255.)
            Hdat.append(int(h * 255.))
            Sdat.append(int(s * 255.))
            Vdat.append(int(v * 255.))
        r.putdata(Hdat)
        g.putdata(Sdat)
        b.putdata(Vdat)
        return Image.merge('RGB', (r, g, b))
    else:
        return None


def rgb2lab(image):
    from colormath.color_objects import sRGBColor, LabColor
    from colormath.color_conversions import convert_color

    out = Image.new('RGB', image.size, 0xffffff)

    width, height = image.size
    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))
            if not r == 0 and not g == 0 and not b == 0:
                rgb = sRGBColor(r, g, b, is_upscaled=True)
                lab = convert_color(rgb, LabColor)
                (l, a, b) = lab.get_value_tuple()
                out.putpixel((x, y), (int(l) + 128, int(a) + 128, int(b) + 128))

    return out


def excess_green(image):
    out = Image.new('RGB', image.size, 0xffffff)

    width, height = image.size
    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))
            out.putpixel((x, y), (2 * g - r - b), 0, 0)


def get_flowers_within_bounds(image_path, x_offset, y_offset, tile_size):
    """
    Returns a list of all annotations that are located within the specified bounds of the image

    Parameters:
        image_path (str): path of the image file
        x_offset (int): the left bound of the tile to search in
        y_offset (int): the top bound of the tile to search in
        tile_size (int): the tile size 
    
    Returns:
        list: a list of dicts containing all annotations within the specified bounds
    """

    filtered_annotations = []
    annotations = file_utils.get_annotations(image_path)

    for flower in annotations:
        if flower["name"] == "roi":
            continue

        [top, left, bottom, right] = flower_info.get_bbox(flower)
        [top, left, bottom, right] = [top - y_offset, left - x_offset, bottom - y_offset, right - x_offset]
        if is_bounding_box_within_image(tile_size, top, left, bottom, right):
            flower["bounding_box"] = [max(-1, top), max(-1, left), min(tile_size + 1, bottom), min(tile_size + 1, right)]
            filtered_annotations.append(flower)

    return filtered_annotations


def is_bounding_box_within_image(tile_size, top, left, bottom, right):
    """Checks if a specified bounding box intersects with the image tile

    Parameters:
        tile_size (int): the tile size 
        top (int): top bound of bounding box 
        left (int): left bound of bounding box 
        bottom (int): bottom bound of bounding box 
        right (int): right bound of bounding box 
    
    Returns:
        bool: True if the specified bounding box is within the image bounds,
            False otherwise
    """

    if left < 0 and right < 0:
        return False
    if left >= tile_size and right >= tile_size:
        return False
    if bottom < 0 and top < 0:
        return False
    if bottom >= tile_size and top >= tile_size:
        return False
    return True


def split_train_dir(src_dir, dst_dir, labels, labels_dst, split_mode, input_folders, splits, full_size_splitted_dir=None):
    """Splits all annotated images into training and testing directory

    Parameters:
        src_dir (str): the directory path containing all images and xml annotation files 
        dst_dir (str): path to the test directory where part of the images (and 
                 annotations) will be copied to
        labels (dict): a dict inside of which the flowers are counted
        labels_dst (dict): a dict inside of which the flowers are counted that
            moved to the dst directory
        split_mode (str): If split_mode is "random", the images are split
            randomly into test and train directory. If split_mode is "deterministic",
            the images will be split in the same way every time this script is 
            executed and therefore making different configurations comparable
        input_folders (list): A list of strings containing all input_folders. 
        splits (list): A list of floats between 0 and 1 of the same length as 
            input_folders. Each boolean indicates what portion of the images
            inside the corresponding input folder should be used for testing or validating and
            not for training
        test_dir_full_size (str): path of folder to which all full size original
            images that are moved to the test directory should be copied to.
            (this folder can be used for evaluation after training) (default is None)
    
    Returns:
        None
    """

    images = file_utils.get_all_images_in_folder(src_dir)

    for input_folder_index in range(0, len(input_folders)):
        portion_to_move_to_test_dir = float(splits[input_folder_index])
        images_in_current_folder = []
        image_names_in_current_folder = []

        # get all image_paths in current folder
        for image_path in images:
            if "inputdir" + str(input_folder_index) in image_path:
                images_in_current_folder.append(image_path)
                image_name = os.path.basename(image_path).split("_subtile", 1)[0]
                if not image_name in image_names_in_current_folder:
                    image_names_in_current_folder.append(image_name)

        if split_mode == "random":
            # shuffle the images randomly
            random.shuffle(images_in_current_folder)
            # and move the first few images to the test folder
            for i in range(0, int(len(images_in_current_folder) * portion_to_move_to_test_dir)):
                move_image_and_annotations_to_folder(images_in_current_folder[i], dst_dir, labels, labels_dst)

        elif split_mode == "deterministic":
            # move every 1/portion_to_move_to_test_dir-th image to the dst_dir
            split_counter = 0.5
            # loop through all image_names (corresponding to untiled original images)
            for image_name in image_names_in_current_folder:
                split_counter = split_counter + portion_to_move_to_test_dir
                # if the split_counter is greater than one, all image_tiles corresponding to that original image should be
                # moved to the test directory
                if split_counter >= 1:
                    if full_size_splitted_dir:
                        original_image_name = os.path.join(input_folders[input_folder_index], image_name)
                        dest_path = os.path.join(full_size_splitted_dir, "inputdir" + str(input_folder_index) + "_" + image_name)
                        copy(original_image_name, dest_path)
                        copy(original_image_name[:-4] + "_annotations.json", os.path.join(full_size_splitted_dir, "inputdir" + str(input_folder_index) + "_" + image_name[:-4] + "_annotations.json"))

                    for image_tile in images_in_current_folder:
                        if image_name in os.path.basename(image_tile):
                            move_image_and_annotations_to_folder(image_tile, dst_dir, labels, labels_dst)
                    split_counter = split_counter - 1


def move_image_and_annotations_to_folder(image_path, dest_folder, labels, labels_test):
    """Moves and image alonside with its annotation file to another folder and updates the flower counts

    Parameters:
        image_path (str): path to the image
        dest_folder (str): path to the destination folder
        labels (dict): a dict inside of which the flowers are counted
        labels_test (dict): a dict inside of which the flowers are counted that
            moved to the test directory
    
    Returns:
        None
    """

    image_name = os.path.basename(image_path)
    xml_name = os.path.basename(image_path)[:-4] + ".xml"

    # update the labels count to represent the counts of the training data
    xmlTree = ET.parse(image_path[:-4] + ".xml")
    for elem in xmlTree.iter():
        if elem.tag == "name":
            flower_name = elem.text
            labels[flower_name] = labels[flower_name] - 1
            add_label_to_labelcount(flower_name, labels_test)
    move(image_path, os.path.join(dest_folder, image_name))
    move(image_path[:-4] + ".xml", os.path.join(dest_folder, xml_name))


def write_labels_to_labelmapfile(labels, output_path):
    """Given a list of labels, prints them to a labelmap file needed by tensorflow

    Parameters:
        labels (dict): a dict inside of which the flowers are counted
        output_path (str): a directory path where the labelmap.pbtxt file should be saved to
    
    Returns:
        None
    """

    output_name = os.path.join(output_path, "label_map.pbtxt")
    end = '\n'
    s = ' '
    out = ''

    for ID, name in enumerate(labels):
        out += 'item' + s + '{' + end
        out += s * 2 + 'id:' + ' ' + (str(ID + 1)) + end
        out += s * 2 + 'name:' + ' ' + '\'' + name + '\'' + end
        out += '}' + end * 3

    with open(output_name, 'w') as f:
        f.write(out)


def build_xml_tree(flowers, image_path, labels):
    """Given a list of flowers, this function builds an xml tree from it. The XML tree is needed to create the tensorflow .record files

    Parameters:
        flowers (list): a list of flower dicts
        image_path (str): the path of the corresponding image
        labels (dict): a dict inside of which the flowers are counted

    Returns:
        None
    """

    root = ET.Element("annotation")

    image = Image.open(image_path)
    ET.SubElement(root, "filename").text = os.path.basename(image_path)

    width, height = image.size
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)

    for flower in flowers:
        flower_name = flower_info.clean_string(flower["name"])
        add_label_to_labelcount(flower_name, labels)

        annotation_object = ET.SubElement(root, "object")
        ET.SubElement(annotation_object, "name").text = flower_name
        ET.SubElement(annotation_object, "pose").text = "Unspecified"
        ET.SubElement(annotation_object, "truncated").text = str(0)
        ET.SubElement(annotation_object, "difficult").text = str(0)
        bndbox = ET.SubElement(annotation_object, "bndbox")

        [top, left, bottom, right] = flower["bounding_box"]
        ET.SubElement(bndbox, "xmin").text = str(left)
        ET.SubElement(bndbox, "ymin").text = str(top)
        ET.SubElement(bndbox, "xmax").text = str(right)
        ET.SubElement(bndbox, "ymax").text = str(bottom)

    tree = ET.ElementTree(root)
    return tree

# small helper function to keep track of how many flowers of each species have been annotated
def add_label_to_labelcount(flower_name, label_count):
    """Small helper function to to add a flower to a label_count dict
    
    Parameters:
        flower_name (str): name of the flower to add to the dict
        label_count (dict): a dict inside of which the flowers are counted

    Returns:
        None
    """

    if label_count.get(flower_name) == None:
        label_count[flower_name] = 1
    else:
        label_count[flower_name] = label_count[flower_name] + 1


def make_training_dir_folder_structure(root_folder):
    """Creates the whole folder structure of the output. All subsequent scripts such as train.py, predict.py, export_inference_graph.py or eval.py rely on this folder structure
    
    Parameters:
        root_folder (str): path to a folder inside of which the project folder structure is created

    Returns:
        None
    """

    images_folder = os.path.join(root_folder, "images")
    os.makedirs(images_folder, exist_ok=True)
    file_utils.delete_folder_contents(images_folder)
    os.makedirs(os.path.join(images_folder, "test"), exist_ok=True)
    os.makedirs(os.path.join(images_folder, "train"), exist_ok=True)
    os.makedirs(os.path.join(images_folder, "test_full_size"), exist_ok=True)
    os.makedirs(os.path.join(images_folder, "validation"), exist_ok=True)
    os.makedirs(os.path.join(images_folder, "validation_full_size"), exist_ok=True)
    prediction_folder = os.path.join(root_folder, "predictions")
    os.makedirs(prediction_folder, exist_ok=True)
    os.makedirs(os.path.join(prediction_folder, "evaluations"), exist_ok=True)
    os.makedirs(os.path.join(root_folder, "model_inputs"), exist_ok=True)
    os.makedirs(os.path.join(root_folder, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(root_folder, "pre-trained-model"), exist_ok=True)
    trained_inference_graphs_folder = os.path.join(root_folder, "trained_inference_graphs")
    os.makedirs(trained_inference_graphs_folder, exist_ok=True)
    training_folder = os.path.join(root_folder, "training")
    os.makedirs(training_folder, exist_ok=True)
    os.makedirs(os.path.join(root_folder, "eval"), exist_ok=True)
    validation_folder = os.path.join(root_folder, "validation")
    os.makedirs(validation_folder, exist_ok=True)
    os.makedirs(os.path.join(validation_folder, "evaluation"), exist_ok=True)


# This function takes the labels map and returns an array with the flower names that have more than min_instances instances
def filter_labels(labels, min_instances=50):
    """Creates an array with the flower names that have more than min_instances instances
    
    Parameters:
        labels (dict): a dict inside of which the flowers are counted
        min_instances (int): minimum instances to pass the filter (default is 50)
    Returns:
        list: a list containing all flower names that have more than min_instances instances
    """

    flowers_to_use = []
    for key, value in labels.items():
        if value >= min_instances:
            flowers_to_use.append(key)
    return flowers_to_use


def set_config_file_parameters(project_dir, num_classes, tensorflow_tile_size):
    """
    Sets various TensorFlow pipeline config parameters, including the number of classes.

    Parameters:
        project_dir (str): project folder path
        num_classes (int): number of classes present in training data
    Returns:
        None
    """
    config_path = os.path.join(project_dir, "pre-trained-model/pipeline.config")
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

    with tf.io.gfile.GFile(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

#Entkommentiere die Parameter des Modells, welches für dsa Training gewählt wurde.

    # Set model parameters for faster rcnn
    pipeline_config.model.faster_rcnn.num_classes = num_classes
    pipeline_config.model.faster_rcnn.image_resizer.fixed_shape_resizer.height = tensorflow_tile_size
    pipeline_config.model.faster_rcnn.image_resizer.fixed_shape_resizer.width = tensorflow_tile_size
    pipeline_config.model.faster_rcnn.first_stage_anchor_generator.grid_anchor_generator.height_stride = 16
    pipeline_config.model.faster_rcnn.first_stage_anchor_generator.grid_anchor_generator.width_stride = 16
    pipeline_config.model.faster_rcnn.first_stage_anchor_generator.grid_anchor_generator.height = 256
    pipeline_config.model.faster_rcnn.first_stage_anchor_generator.grid_anchor_generator.width = 256
    pipeline_config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_detections_per_class = 300
    pipeline_config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_total_detections = 300
    pipeline_config.model.faster_rcnn.first_stage_max_proposals = 300
    pipeline_config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.score_threshold = 0.0

    #Set model parameters for SSD or EfficientDet
    #pipeline_config.model.ssd.num_classes = num_classes
    #pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height = tensorflow_tile_size
    #pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.width = tensorflow_tile_size
    #pipeline_config.model.ssd.anchor_generator.multiscale_anchor_generator.scales_per_octave = 4
    #pipeline_config.model.ssd.post_processing.batch_non_max_suppression.max_detections_per_class = 300
    #pipeline_config.model.ssd.post_processing.batch_non_max_suppression.max_total_detections = 300
    #pipeline_config.model.ssd.post_processing.batch_non_max_suppression.score_threshold = 0.0

    pipeline_config.train_config.batch_size = 8
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_config.use_bfloat16 = False

    # Set fine-tune checkpoint
    pre_trained_model_folder = os.path.join(project_dir, "pre-trained-model")
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(pre_trained_model_folder, "checkpoint/ckpt-0")

    # Set train input reader
    model_inputs_folder = os.path.join(project_dir, "model_inputs")
    pipeline_config.train_input_reader.label_map_path = os.path.join(model_inputs_folder, "label_map.pbtxt")
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(model_inputs_folder, "train.record")]

    # Set eval input reader
    pipeline_config.eval_input_reader[0].label_map_path = os.path.join(model_inputs_folder, "label_map.pbtxt")
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(model_inputs_folder, "test.record")]

# Set data augmentation options
    if constants.data_augmentation_enabled:
    # Check and add missing augmentations
        augmentation_options = pipeline_config.train_config.data_augmentation_options

    # Add random vertical flip if not present
        if not any(opt.HasField('random_vertical_flip') for opt in augmentation_options):
            d1 = pipeline_config.train_config.data_augmentation_options.add()
            d1.random_vertical_flip.CopyFrom(preprocessor_pb2.RandomVerticalFlip())

    # Add random horizontal flip if not present
        if not any(opt.HasField('random_horizontal_flip') for opt in augmentation_options):
            d1 = pipeline_config.train_config.data_augmentation_options.add()
            d1.random_horizontal_flip.CopyFrom(preprocessor_pb2.RandomHorizontalFlip())

    # Add random adjust brightness if not present
        if not any(opt.HasField('random_adjust_brightness') for opt in augmentation_options):
            d1 = pipeline_config.train_config.data_augmentation_options.add()
            d1.random_adjust_brightness.CopyFrom(preprocessor_pb2.RandomAdjustBrightness())

    # Add random adjust contrast if not present
        if not any(opt.HasField('random_adjust_contrast') for opt in augmentation_options):
            d1 = pipeline_config.train_config.data_augmentation_options.add()
            d1.random_adjust_contrast.CopyFrom(preprocessor_pb2.RandomAdjustContrast())

    # Add random adjust saturation if not present
        if not any(opt.HasField('random_adjust_saturation') for opt in augmentation_options):
            d1 = pipeline_config.train_config.data_augmentation_options.add()
            d1.random_adjust_saturation.CopyFrom(preprocessor_pb2.RandomAdjustSaturation())

    # Add random jitter boxes if not present
        if not any(opt.HasField('random_jitter_boxes') for opt in augmentation_options):
            d1 = pipeline_config.train_config.data_augmentation_options.add()
            d1.random_jitter_boxes.CopyFrom(preprocessor_pb2.RandomJitterBoxes())

    # Add random adjust hue if not present (if it is in the default config)
        if not any(opt.HasField('random_adjust_hue') for opt in augmentation_options):
            d1 = pipeline_config.train_config.data_augmentation_options.add()
            d1.random_adjust_hue.CopyFrom(preprocessor_pb2.RandomAdjustHue())


    # Write the updated config back to the file
    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(config_path, "wb") as f:
        f.write(config_text)

    print("Pipeline configuration updated successfully.")


def print_labels(labels, flowers_to_use):
    """Prints the label_count dict to the console in readable format
    
    Parameters:
        labels (dict): a dict inside of which the flowers are counted
        flowers_to_use (list): list of strings of the flower names that should
            be used for training
    Returns:
        None
    """

    print("used:")
    for key, value in labels.items():
        if key in flowers_to_use:
            print("    " + key + ": " + str(value))

    print("not used due to lack of enough annotation data:")
    for key, value in labels.items():
        if not (key in flowers_to_use):
            print("    " + key + ": " + str(value))


pbar = None


def download_pretrained_model(project_folder, link):
    """
    Downloads the pretrained model from the provided link and unpacks it into
    the pre-trained-model folder.
    
    Parameters:
        project_folder (str): the project folder path
        link (str): download link of the model
    Returns:
        None
    """

    pretrained_model_folder = os.path.join(project_folder, "pre-trained-model")

    destination_file = os.path.join(pretrained_model_folder, "downloaded_model.tar.gz")
    if not os.path.isfile(destination_file):

        def show_progress(block_num, block_size, total_size):
            global pbar
            if pbar is None:
                pbar = progressbar.ProgressBar(maxval=total_size)

            downloaded = block_num * block_size
            if downloaded < total_size:
                pbar.update(downloaded)
            else:
                pbar.finish()
                pbar = None

        print("Downloading pretrained model...")
        # Download the file from `url` and save it locally under `file_name`:
        urllib.request.urlretrieve(link, destination_file, show_progress)

        tf = tarfile.open(destination_file)
        tf.extractall(pretrained_model_folder)

        for folder in os.listdir(pretrained_model_folder):
            folder = os.path.join(pretrained_model_folder, folder)
            if os.path.isdir(folder):
                for file_to_move in os.listdir(folder):
                    file_to_move = os.path.join(folder, file_to_move)
                    shutil.move(file_to_move, os.path.join(pretrained_model_folder, os.path.basename(file_to_move)))
                os.rmdir(folder)
        
   
  
if __name__== "__main__":

    input_folders = constants.input_folders
    test_splits = constants.test_splits
    validation_splits = constants.validation_splits
    output_dir = constants.train_dir
    tile_size = constants.tile_size
    split_mode = constants.split_mode
    min_flowers = constants.min_flowers
    overlap=constants.train_overlap
    model_link=constants.pretrained_model_link
    tensorflow_tile_size=constants.tensorflow_tile_size
    
    convert_annotation_folders(input_folders, test_splits, validation_splits, output_dir, tile_size, split_mode, min_flowers, overlap, model_link, tensorflow_tile_size)