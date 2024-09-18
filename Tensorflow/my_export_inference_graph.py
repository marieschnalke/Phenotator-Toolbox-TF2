"""

This script was largely adapted from the Tensorflow Object Detection repository,
Only slight adjustions were made to integrate it into the command line tool.

"""


import os
from utils import file_utils
from utils import constants

import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter_lib_v2 as exporter
from object_detection.protos import pipeline_pb2
import tf_slim as slim
from importlib import reload  # Python 3.4+ only.

def main(pipeline_config_path, trained_checkpoint_prefix, output_directory):
    """
    Main function executing the export. This is directly adapted from the Tensorflow implementation.
    """
    config_override = ''
    input_type = 'image_tensor'
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    
    with tf.io.gfile.GFile(pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    
    text_format.Merge(config_override, pipeline_config)
    reload(exporter)
    
    exporter.export_inference_graph(
        input_type=input_type,
        pipeline_config=pipeline_config,
        trained_checkpoint_dir=os.path.dirname(trained_checkpoint_prefix),
        output_directory=output_directory
    )


def find_best_model(training_directory, look_in_checkpoints_dir=False, model_selection_criterion="f1"):
    """
    Finds the best model. If look_in_checkpoints_dir is set to True, the model best model is
    selected based on the score it achieved on the validation set using either the f1 score or
    mAP. 

    Parameters:
        training_directory (str): path to the training directory
        look_in_checkpoints_dir (bool): if True, it was trained with a validation set
            and the best model according to the model_selection_criterion is chosen
        model_selection_criterion (str): either 'f1' or 'mAP'. 
    
    Returns:
        str: path of the best model prefix or None if no valid checkpoint is found
    """
    largest_number = -1
    best_checkpoint = None
    
    for file in os.listdir(training_directory):
        if file.endswith(".index") and file.startswith("ckpt-"):
            start = file.index("ckpt-") + len("ckpt-")
            end = file.index(".index", start)
            curr_number = int(file[start:end])
            if curr_number > largest_number:
                largest_number = curr_number
                best_checkpoint = os.path.join(training_directory, f"ckpt-{largest_number}")
    
    if best_checkpoint:
        print(f"Found last checkpoint: {best_checkpoint}")
        return best_checkpoint
    
    print("No valid checkpoints found in the training directory.")
    return None

def run(project_dir, look_in_checkpoints_dir=False, model_selection_criterion="f1", checkpoint=None):
    """
    Runs the export command.
    
    Parameters:
        project_dir (str): path to the project directory
        look_in_checkpoints_dir (bool): if True, it was trained with a validation set
            and the best model according to the model_selection_criterion is chosen
        model_selection_criterion (str): either 'f1' or 'mAP'. 
    
    Returns:
        None
    """
    train_dir = project_dir
    output_directory = os.path.join(train_dir, "trained_inference_graphs")
    pipeline_config_path = os.path.join(train_dir, "pre-trained-model", "pipeline.config")
    training_directory = os.path.join(project_dir, "training")
    
    if checkpoint:
        trained_checkpoint_prefix = os.path.join(training_directory, f"ckpt-{checkpoint}")
    else:
        trained_checkpoint_prefix = find_best_model(training_directory, look_in_checkpoints_dir, model_selection_criterion)
    
    if trained_checkpoint_prefix is None:
        raise ValueError("No valid checkpoint found. Please check your training directory.")
    
    print("Exporting " + trained_checkpoint_prefix)

    file_utils.delete_folder_contents(output_directory)
    
    main(pipeline_config_path, trained_checkpoint_prefix, output_directory)
    print("Done exporting " + trained_checkpoint_prefix)

if __name__ == '__main__':
    print("Please use the export-inference-graph command in the cli.py script to execute this script.")
