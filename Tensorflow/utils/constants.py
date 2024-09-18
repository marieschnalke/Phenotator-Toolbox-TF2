# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:17:19 2019

@author: gallmanj
@updated_by: mschnalke

The script was originally authored by gallmanj and has been adapted to newer models and software packages.
"""

'''MOST IMPORTANT SETTINGS'''
project_folder = "..."

#entkommentiere das gewollte Modell

pretrained_model_link = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8.tar.gz"   #Faster R-CNN
#pretrained_model_link = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz"       #SSD
#pretrained_model_link = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz"                     #EfficientDet



'''image-preprocessing command parameters'''
input_folders = ["...",
                 "...",
                 "...",
                 "...",
                 "...",
                 "...",
                 "...",
                 "...",
                 "..."]


test_splits = [0.2,
                        0.2,
                        0.2,
                        0.2,
                        0.2,
                        0.2,
                        0.2,
                        0.2,
                        0.2]
                        

validation_splits = [0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1]

split_mode = "deterministic"
train_tile_sizes =[512]
train_overlap = 50
min_flowers = 50 #minimum amount of flower instances to include species in training
#All images will be resized to tensorflow_tile_size x tensorflow_tile_size tiles
#choose a smaller tensorflow_tile_size if your gpu is not powerful enough to handle
#900 x 900 pixel tiles
tensorflow_tile_size = 1024
data_augmentation_enabled = True

'''train command parameters'''
max_steps = 200000
with_validation = True
model_selection_criterion = "f1" #also used for export-inference-graph command

'''predict command parameters'''
images_to_predict = project_folder + "/images/test_full_size"
predictions_folder = project_folder + "/predictions"
prediction_tile_size = 512
prediction_overlap = 50
min_confidence_score = 0.2 #also used by evaluate command
visualize_predictions = True
visualize_groundtruth = False
visualize_name = False
visualize_score = False
max_iou = 0.3


'''evaluate command parameters'''
prediction_evaluation_folder = predictions_folder + "/evaluations"
iou_threshold = 0.3
generate_visualizations = False
print_confusion_matrix = False
visualize_info = True

'''visualization command parameters'''
visualize_bounding_boxes_with_name = True
clean_output_folder = True

'''copy-annotations command parameters'''
one_by_one = True

'''prepare-for-tablet command parameters'''
prepare_for_tablet_tile_size = 256

'''generate-heatmap command parameters'''
heatmap_width=100
max_val = None
classes = None
overlay = True
output_image_width = 1000

'''Set the radius (not diameter) of each flower species'''
flower_bounding_box_size = {

'achillea millefolium': 20,
'anthyllis vulneraria'   : 16,
'agrimonia eupatoria': 15,
'carum carvi'   : 22,
'centaurea jacea': 14,
'cerastium caespitosum'   : 7,
'crepis biennis'   : 17,
'daucus carota': 23,
'galium mollugo'   : 4,
'knautia arvensis'   : 17,
'medicago lupulina'   : 4,
'leucanthemum vulgare'   : 20,
'lotus corniculatus'   : 8,
'lychnis flos cuculi'   : 15,
'myosotis arvensis'   : 6,
'onobrychis viciifolia'   : 10,
'picris hieracioides': 13,
'plantago lanceolata'   : 8,
'plantago major'   : 11,
'prunella vulgaris': 10,
'ranunculus acris'   : 11,
'ranunculus bulbosus'   : 11,
'ranunculus friesianus'   : 11,
'ranunculus'   : 11,
'salvia pratensis'   : 15,
'tragopogon pratensis'   : 17,
'trifolium pratense'   : 10,
'veronica chamaedris'   : 4,
'vicia sativa'   : 6,
'vicia sepium'   : 4,
'dianthus carthusianorum': 11,
'lathyrus pratensis' : 8,
'leontodon hispidus' : 18,
'rhinanthus alectorolophus': 20,
'trifolium repens': 10,
'orchis sp': 20
}
flower_groups = {
        
    "ranunculus": ["ranunculus bulbosus", "ranunculus friesianus", "ranunculus acris"],
    "lotus corniculatus": ["lotus corniculatus", "lathyrus pratensis"],
    "galium mollugo": ["galium mollugo","carum carvi", "achillea millefolium", "daucus carota"],
    "crepis biennis": ["crepis biennis", "leontodon hispidus", "tragopogon pratensis", "picris hieracioides"],
    "centaurea jacea": ["centaurea jacea", "lychnis flos cuculi"]           
}



