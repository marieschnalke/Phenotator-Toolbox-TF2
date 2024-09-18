# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 20:42:09 2019

@author: johan

@updated_by: mschnalke

The script was originally authored by gallmanj and has been adapted to newer models and software packages.

This script lets the user edit annotations inside the LabelMe application.

"""


import tensorflow as tf
import os
from shutil import copyfile, rmtree, copytree
import train
import my_export_inference_graph
import custom_evaluations
import predict
from utils import constants

def train_with_validation(project_dir, max_steps):
    """
    Trains a network using a validation set.
    The prediction and evaluation algorithms are run on the validation set every 2500 steps.

    Parameters:
        project_dir (str): The project directory used during the image-preprocessing command.
        max_steps (int): The network is trained for at most max_steps steps.

    Returns:
        None
    """

    images_folder = os.path.join(project_dir, "images")
    validation_images_folder = os.path.join(images_folder, "validation")
    validation_folder = os.path.join(project_dir, "validation")
    evaluation_folder = os.path.join(validation_folder, "evaluation")
    training_folder = os.path.join(project_dir, "training")
    checkpoints_folder = os.path.join(training_folder, "checkpoints")
    best_model_folder = os.path.join(project_dir, "best_model")
    
    os.makedirs(checkpoints_folder, exist_ok=True)
    os.makedirs(best_model_folder, exist_ok=True)

    precision_recall_file = os.path.join(checkpoints_folder, "precision_recall_evolution.txt")
    
    current_checkpoint = get_max_checkpoint(training_folder)

    if current_checkpoint == -1:
        current_step = 0
        print("Kein vorheriger Checkpoint gefunden. Starte Training bei Schritt 0.")
    else:
        current_step = (current_checkpoint - 1) * 2500  # Korrektur: Ziehe 1 vom Checkpoint ab
        print(f"Fortsetzung des Trainings ab Checkpoint {current_checkpoint} mit {current_step} Schritten.")

    # Lade den besten F1-Score aus der Datei, wenn vorhanden
    best_f1 = load_best_f1_from_file(precision_recall_file)

    for num_steps in range(current_step + 2500, max_steps + 2500, 2500):
        print(f"Trainiere bis Schritt {num_steps}...")
        train.run(project_dir, num_steps)
        
        print(f"Checkpoint {num_steps} wird exportiert und evaluiert...")
        copy_checkpoint_to_folder(num_steps, training_folder, checkpoints_folder)
        my_export_inference_graph.run(project_dir, look_in_checkpoints_dir=False)
        
        predict.predict(project_dir, validation_images_folder, validation_folder, constants.prediction_tile_size, constants.prediction_overlap)
        
        stats = custom_evaluations.evaluate(project_dir, validation_folder, evaluation_folder)["overall"]
        (precision, recall, mAP, f1) = get_precision_and_recall_from_stat(stats)
        
        with open(precision_recall_file, "a") as text_file:
            text_file.write(f"step {num_steps}; precision: {precision} recall: {recall} mAP: {mAP} f1: {f1}\n")
        
        print(f"Evaluierung für Schritt {num_steps} abgeschlossen. Precision: {precision}, Recall: {recall}, mAP: {mAP}, F1: {f1}")
        
        if f1 >= best_f1:
            best_f1 = f1
            print(f"Neues bestes Modell gefunden mit F1-Score {f1} bei Schritt {num_steps}. Speichere Modell...")
            
            # Kopiere den letzten Checkpoint, das SavedModel und die Pipeline-Konfiguration
            latest_checkpoint = get_max_checkpoint(training_folder)
            copy_model_to_best_folder(latest_checkpoint, training_folder, best_model_folder)

def load_best_f1_from_file(precision_recall_file):
    """
    Lädt den besten F1-Score aus der precision_recall_evolution.txt-Datei.

    Parameters:
        precision_recall_file (str): Pfad zur precision_recall_evolution.txt-Datei

    Returns:
        float: Der beste F1-Score, der bisher aufgezeichnet wurde
    """
    best_f1 = 0
    if os.path.exists(precision_recall_file):
        with open(precision_recall_file, "r") as file:
            for line in file:
                if "f1:" in line:
                    f1_score = float(line.strip().split("f1:")[-1])
                    if f1_score > best_f1:
                        best_f1 = f1_score
    return best_f1

def copy_checkpoint_to_folder(checkpoint, src_dir, dst_dir):
    """
    Kopiert die Checkpoint-Dateien von einem Ordner in einen anderen.
    
    Parameters:
        checkpoint (int): Eindeutige Identifikationsnummer für den Checkpoint.
        src_dir (str): Quellordner, der die Checkpoint-Dateien enthält.
        dst_dir (str): Zielordner, in den die Checkpoint-Dateien kopiert werden sollen.
        
    Returns:
        None
    """
    for file in os.listdir(src_dir):
        if f"ckpt-{checkpoint}" in file:
            copyfile(os.path.join(src_dir, file), os.path.join(dst_dir, file))

def copy_model_to_best_folder(checkpoint, src_dir, dst_dir):
    """
    Kopiert das SavedModel, die Checkpoints und die pipeline.config in das best_model-Verzeichnis.

    Parameters:
        checkpoint (int): Einzigartige Identifikationsnummer für den Checkpoint
        src_dir (str): Quellverzeichnis, das die Checkpoint-Dateien enthält
        dst_dir (str): Zielverzeichnis, in das das beste Modell gespeichert werden soll

    Returns:
        None
    """
    if os.path.exists(dst_dir):
        rmtree(dst_dir)
    os.makedirs(dst_dir)

    # Kopiere Checkpoint-Dateien
    checkpoint_prefix = f"ckpt-{checkpoint}"
    for file in os.listdir(src_dir):
        if file.startswith(checkpoint_prefix):
            copyfile(os.path.join(src_dir, file), os.path.join(dst_dir, file))

    # Kopiere SavedModel aus dem trained_inference_graphs-Verzeichnis
    trained_inference_graphs_dir = os.path.join(os.path.dirname(src_dir), "trained_inference_graphs", "saved_model")
    saved_model_dst = os.path.join(dst_dir, "saved_model")
    if os.path.exists(trained_inference_graphs_dir):
        copytree(trained_inference_graphs_dir, saved_model_dst)

    # Kopiere Pipeline-Konfiguration
    pipeline_config_src = os.path.join(src_dir, "pipeline.config")
    pipeline_config_dst = os.path.join(dst_dir, "pipeline.config")
    if os.path.exists(pipeline_config_src):
        copyfile(pipeline_config_src, pipeline_config_dst)

def get_max_checkpoint(training_folder):
    """
    Finds the most advanced training checkpoint saved by TensorFlow within a folder
    
    Parameters:
        training_folder (str): path to the folder containing the checkpoints
    Returns:
        int: the unique identifier number of the checkpoint file
    """

    largest_number = -1                        
    for file in os.listdir(training_folder):
        if file.endswith(".index") and file.startswith("ckpt-"):
            start = file.index("ckpt-") + len("ckpt-")
            end = file.index(".index", start)
            curr_number = int(file[start:end])
            if curr_number > largest_number:
                largest_number = curr_number
    return largest_number

def get_precision_and_recall_from_stat(stat):
    """
    Extracts the precision/recall/mAP/f1 scores from the stat dict that is returned
    by the evaluation script.
    
    Parameters:
        stat (dict): dict returned by the evaluation script
        
    Returns:
        tuple: (precision, recall, mAP, f1)
    """
    n = stat["tp"] + stat["fn"]
    if float(stat["fp"] + stat["tp"]) == 0:
        precision = "0"
    else:
        precision = float(stat["tp"]) / float(stat["fp"] + stat["tp"])
    if float(stat["tp"] + stat["fn"]) == 0:
        recall = "0"
    else:
        recall = float(stat["tp"]) / float(stat["tp"] + stat["fn"])

    f1 = 0
    if recall != "0" and precision != "0" and ((recall > 0) or (precision > 0)):
        f1 = 2 * (precision * recall) / (precision + recall)

    return (float(precision), float(recall), stat["mAP"], f1)

if __name__ == '__main__':
    train_with_validation(constants.project_dir, constants.max_steps)
