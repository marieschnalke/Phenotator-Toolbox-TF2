# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


'''
This script trains a model. 
'''

print("Loading libraries...")
import functools
import json
import os
import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from shutil import copyfile
from utils import constants

from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection import model_lib_v2

tf.get_logger().setLevel('INFO')

def main(train_dir, pipeline_config_path):
    """
    Main function adapted for TensorFlow 2.
    Trains a network. Saves checkpoints and other files within the train_dir and
    trains according to the parameters given in the config file located at the
    pipeline_config_path.
    
    Parameters:
        train_dir (str): path of the directory where the checkpoints should be saved to
        pipeline_config_path (str): path to the pipeline.config file
        
    Returns:
        None
    """
    tf.io.gfile.makedirs(train_dir)
    if pipeline_config_path:
        configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
        tf.io.gfile.copy(pipeline_config_path, os.path.join(train_dir, 'pipeline.config'), overwrite=True)
    else:
        raise ValueError('pipeline_config_path is required.')

    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']

    model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=True)

    def get_next(config):
    # Build the dataset using the Object Detection API's dataset builder
        dataset = dataset_builder.build(config)
    # The dataset builder should handle parsing internally
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(config.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset



    train_input_fn = functools.partial(get_next, input_config)

    # Use TF2's distributed training
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model_lib_v2.train_loop(
            pipeline_config_path=pipeline_config_path,
            model_dir=train_dir,
            train_steps=train_config.num_steps,
            use_tpu=False,
            checkpoint_every_n=2500,
            record_summaries=True)

def set_num_steps_in_config_file(num_steps, project_dir):
    """
    Saves the num_steps value to the config file that is used by the training process.

    Parameters:
        num_steps (int): number of steps to train the network
        project_dir (str): the project folder path created with the image-preprocessing
            script.
    
    Returns:
        None
    """
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()                                                                                                                                                                                                          
    with tf.io.gfile.GFile(project_dir + "/pre-trained-model/pipeline.config", "r") as f:                                                                                                                                                                                                                     
        proto_str = f.read()                                                                                                                                                                                                                                          
        text_format.Merge(proto_str, pipeline_config)                                                                                                                                                                                                                 

    pipeline_config.train_config.num_steps = num_steps                                                                                                                                                                                          

    config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
    with tf.io.gfile.GFile(project_dir + "/pre-trained-model/pipeline.config", "wb") as f:                                                                                                                                                                                                                       
        f.write(config_text)   
    
    # Copy file to training folder
    src = project_dir + "/pre-trained-model/pipeline.config"
    dst = project_dir + "/training/pipeline.config"
    copyfile(src, dst)
                                                                                                                                                                                                                                       
def run(project_dir, max_steps):
    """
    Runs the train command for at most max_steps steps.

    Parameters:
        project_dir (str): the project folder path created with the image-preprocessing
            script.
        max_steps (int): max number of steps to train the network
    
    Returns:
        None
    """
    set_num_steps_in_config_file(max_steps, project_dir)
    pipeline_config_path = project_dir + "/pre-trained-model/pipeline.config"
    
    training_dir = os.path.join(project_dir, "training")
    
    main(training_dir, pipeline_config_path)

if __name__ == '__main__':
    train_dir = constants.project_folder
    run(train_dir, constants.max_steps)
