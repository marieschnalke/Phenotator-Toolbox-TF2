# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:47:15 2019

@author: johan
@updated_by: mschnalke

The script was originally authored by gallmanj and has been adapted to newer models and software packages.


This script converts the xml annotations created by the image-preprocessing
script into csv annotations which can then be converted into tf record files
by the generate_tfrecord script.
"""

import glob
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
import os
import argparse
import io
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple


def xml_to_csv(input_folder, output_path, flowers_to_use=None):
    """Iterates through all .xml files in the input_folder and combines them in a single Pandas dataframe.

    Parameters:
        input_folder (str): path to the input folder containing all images and
            xml annotation files
        output_path (str): path to the output csv file.
        flowers_to_use (list): a list of strings containing flower names. Only
            the annotations with flowernames present in the flowers_to_use list
            are copied to the output csv file. If flowers_to_use is None, all
            annotations are used.
    
    Returns:
        Pandas dataframe of the csv list.
    """

    xml_list = []
    for xml_file in glob.glob(input_folder + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            if flowers_to_use is None or member[0].text in flowers_to_use:
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                         )
                xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv(output_path, index=None)
    return xml_df


def class_text_to_int(row_label, label_map_dict):
    return label_map_dict[row_label]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, label_map_dict):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class'], label_map_dict))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def xml_to_tfrecord(xml_dir, labels_path, output_path, image_dir=None, csv_path=None):
    if image_dir is None:
        image_dir = xml_dir

    label_map = label_map_util.load_labelmap(labels_path)
    label_map_dict = label_map_util.get_label_map_dict(label_map)

    writer = tf.io.TFRecordWriter(output_path)
    path = os.path.join(image_dir)
    examples = xml_to_csv(xml_dir, csv_path)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path, label_map_dict)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecord file: {}'.format(output_path))
    if csv_path is not None:
        examples.to_csv(csv_path, index=None)
        print('Successfully created the CSV file: {}'.format(csv_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample TensorFlow XML-to-TFRecord converter")
    parser.add_argument("-x", "--xml_dir", help="Path to the folder where the input .xml files are stored.", type=str)
    parser.add_argument("-l", "--labels_path", help="Path to the labels (.pbtxt) file.", type=str)
    parser.add_argument("-o", "--output_path", help="Path of output TFRecord (.record) file.", type=str)
    parser.add_argument("-i", "--image_dir", help="Path to the folder where the input image files are stored. Defaults to the same directory as XML_DIR.", type=str, default=None)
    parser.add_argument("-c", "--csv_path", help="Path of output .csv file. If none provided, then no file will be written.", type=str, default=None)

    args = parser.parse_args()

    xml_to_tfrecord(args.xml_dir, args.labels_path, args.output_path, args.image_dir, args.csv_path)
