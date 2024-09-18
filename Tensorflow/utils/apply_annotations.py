# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:48:04 2019

@author: gallmanj
@updated_by: mschnalke

The script was originally authored by gallmanj and has been adapted to newer models and software packages.
"""




import sys
from osgeo import gdal
import os
import pyproj
import json
from PIL import Image
from utils import file_utils
import progressbar
from shapely.geometry import Polygon, box
from osgeo import osr
import numpy as np

class GeoInformation(object):
    def __init__(self, dictionary=None):
        if dictionary:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        else:
            self.lr_lon = 0
            self.lr_lat = 0
            self.ul_lon = 0
            self.ul_lat = 0

def apply_annotations_to_images(annotated_folder, images_folder, output_folder):
    all_ortho_tifs = file_utils.get_all_images_in_folder(images_folder)
    print("Adding Annotations to all ortho images:")
    sys.stdout.flush()

    for i in progressbar.progressbar(range(len(all_ortho_tifs))):
        print(f"Processing image {i+1}/{len(all_ortho_tifs)}: {all_ortho_tifs[i]}")
        sys.stdout.flush()
        apply_annotations_to_image(annotated_folder, all_ortho_tifs[i], output_folder)

def apply_annotations_to_image(annotated_folder, image_path, output_folder):
    all_annotated_images = file_utils.get_all_images_in_folder(annotated_folder)
    ortho_tif = image_path

    try:
        im = Image.open(ortho_tif)
        im.thumbnail(im.size)
        ortho_png = os.path.join(output_folder, os.path.basename(ortho_tif)[:-4] + ".png")
        im.save(ortho_png, quality=100)

        annotation_template = {"annotatedFlowers": []}
        with open(os.path.join(output_folder, os.path.basename(ortho_tif)[:-4] + "_annotations.json"), 'w') as outfile:
            json.dump(annotation_template, outfile)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.stdout.flush()
        return

    c = get_geo_coordinates(ortho_tif)
    print(f"Geo coordinates for {image_path}: {c}")
    sys.stdout.flush()

    if not c:
        print(f"Geo-coordinates not found for image: {ortho_tif}")
        sys.stdout.flush()
        return

    for annotated_image in all_annotated_images:
        d = get_geo_coordinates(annotated_image)
        intersection = get_intersection(c, d)
        if intersection:
            copy_annotations(annotated_image, ortho_png, c, d)

def copy_annotations(annotated_image_path, to_be_annotated_image, to_be_annotated_image_coordinates, annotated_image_coordinates):
    print("Entered copy_annotations")
    sys.stdout.flush()

    annotations = file_utils.get_annotations(annotated_image_path)
    if annotations:
        print(f"Annotations for {annotated_image_path}: {annotations}")
        sys.stdout.flush()
    else:
        print(f"No annotations found for {annotated_image_path}")
        sys.stdout.flush()
        return

    image = Image.open(annotated_image_path)
    width, height = image.size
    print(f"Image size: {width}x{height}")
    sys.stdout.flush()

    orthoTif = Image.open(to_be_annotated_image)
    ortho_width, ortho_height = orthoTif.size
    print(f"Ortho image size: {ortho_width}x{ortho_height}")
    sys.stdout.flush()

    output_annotations_path = to_be_annotated_image[:-4] + "_annotations.json"
    output_annotations = file_utils.read_json_file(output_annotations_path) or {"annotatedFlowers": []}

    for annotation in annotations:
        print(f"Processing annotation: {annotation}")
        sys.stdout.flush()

        should_be_added = True
        translated_annotation = translate_annotation(annotation, height, width, annotated_image_coordinates, to_be_annotated_image_coordinates, ortho_height, ortho_width)
        print(f"Translated annotation: {translated_annotation}")
        sys.stdout.flush()

        if annotation["name"] == "roi":
            result_polygon = get_intersection_of_polygon_and_image_bounds(ortho_width, ortho_height, annotation["polygon"])
            if result_polygon:
                translated_annotation["polygon"] = result_polygon
                output_annotations["annotatedFlowers"].append(translated_annotation)
            continue

        for coord in translated_annotation["polygon"]:
            x, y = coord["x"], coord["y"]
            print(f"Checking coordinates {x}, {y}")
            sys.stdout.flush()

            if not are_coordinates_within_image_bounds(x, y, ortho_width, ortho_height):
                print(f"Coordinates out of bounds: {x}, {y}")
                sys.stdout.flush()
                should_be_added = False
                break
            elif is_pixel_white(x, y, orthoTif):
                print(f"Pixel at {x}, {y} is white")
                sys.stdout.flush()
                should_be_added = False
                break

        if should_be_added:
            output_annotations["annotatedFlowers"].append(translated_annotation)

    # Heatmap-Inkrementierung debuggen
    heatmaps = {"overall": np.zeros((ortho_height, ortho_width))}
    for heatmap_y in range(ortho_height):
        for heatmap_x in range(ortho_width):
            heatmaps["overall"][heatmap_y][heatmap_x] += 1
            print(f"Updated heatmap at ({heatmap_x},{heatmap_y}) with value: {heatmaps['overall'][heatmap_y][heatmap_x]}")
            sys.stdout.flush()

    # Speichere Heatmap für späteren Einsatz (dieser Teil ist nur zu Debugging-Zwecken)
    np.save(output_annotations_path[:-4] + "_heatmap.npy", heatmaps["overall"])

def translate_annotation(annotation, height, width, annotated_image_coordinates, ortho_tif_coordinates, ortho_height, ortho_width):
    output_annotation = annotation
    for coord in output_annotation["polygon"]:
        x, y = coord["x"], coord["y"]
        x_target, y_target = translate_pixel_coordinates(x, y, height, width, annotated_image_coordinates, ortho_tif_coordinates, ortho_height, ortho_width)
        coord["x"], coord["y"] = x_target, y_target

    return output_annotation

def get_intersection_of_polygon_and_image_bounds(image_width, image_height, roi_polygon):
    image_box = box(0, 0, image_width, image_height)
    roi_polygon_array = [(coord["x"], coord["y"]) for coord in roi_polygon]
    roi_polygon = Polygon(roi_polygon_array)
    intersection = image_box.intersection(roi_polygon)

    if intersection.is_empty:
        return None

    return [{"x": x, "y": y} for x, y in intersection.exterior.coords[:-1]]

def are_coordinates_within_image_bounds(x, y, width, height):
    return 0 <= x < width and 0 <= y < height

def is_pixel_white(x, y, image):
    if not are_coordinates_within_image_bounds(x, y, image.width, image.height):
        return False
    return image.load()[x, y] == (255, 255, 255)

def translate_pixel_coordinates(x, y, height, width, source_geo_coords, target_geo_coords, height_target, width_target):
    rel_x = x / width
    rel_y = y / height

    if not source_geo_coords or not target_geo_coords:
        return rel_x * width_target, rel_y * height_target

    geo_x = (source_geo_coords.lr_lon - source_geo_coords.ul_lon) * rel_x + source_geo_coords.ul_lon
    geo_y = (source_geo_coords.ul_lat - source_geo_coords.lr_lat) * (1 - rel_y) + source_geo_coords.lr_lat

    rel_x_target = (geo_x - target_geo_coords.ul_lon) / (target_geo_coords.lr_lon - target_geo_coords.ul_lon)
    rel_y_target = 1 - (geo_y - target_geo_coords.lr_lat) / (target_geo_coords.ul_lat - target_geo_coords.lr_lat)

    print(f"Source X,Y: {x},{y} -> Target X,Y: {rel_x_target * width_target},{rel_y_target * height_target}")
    sys.stdout.flush()

    return rel_x_target * width_target, rel_y_target * height_target

def get_geo_coordinates(input_image):
    if input_image.endswith((".png", ".jpg")):
        try:
            geo_info_path = input_image[:-4] + "_geoinfo.json"
            with open(geo_info_path, 'r') as f:
                datastore = json.load(f)
                geo_info = GeoInformation(datastore)
                swiss = pyproj.Proj("+init=EPSG:2056")
                wgs84 = pyproj.Proj("+init=EPSG:4326")
                geo_info.lr_lon, geo_info.lr_lat = pyproj.transform(wgs84, swiss, geo_info.lr_lon, geo_info.lr_lat)
                geo_info.ul_lon, geo_info.ul_lat = pyproj.transform(wgs84, swiss, geo_info.ul_lon, geo_info.ul_lat)
                return geo_info
        except FileNotFoundError:
            return None

    else:
        try:
            ds = gdal.Open(input_image)
            inSRS_converter = osr.SpatialReference()
            inSRS_converter.ImportFromWkt(ds.GetProjection())
            inSRS_forPyProj = inSRS_converter.ExportToProj4()

            input_coord_system = pyproj.Proj(inSRS_forPyProj)
            swiss = pyproj.Proj("+init=EPSG:2056")

            ulx, xres, xskew, uly, yskew, yres = ds.GetGeoTransform()
            lrx = ulx + (ds.RasterXSize * xres)
            lry = uly + (ds.RasterYSize * yres)

            geo_info = GeoInformation()
            geo_info.lr_lon, geo_info.lr_lat = pyproj.transform(input_coord_system, swiss, lrx, lry)
            geo_info.ul_lon, geo_info.ul_lat = pyproj.transform(input_coord_system, swiss, ulx, uly)

            return geo_info
        except RuntimeError:
            return None

def get_intersection(geo1, geo2):
    leftX = max(geo1.ul_lon, geo2.ul_lon)
    rightX = min(geo1.lr_lon, geo2.lr_lon)
    topY = min(geo1.ul_lat, geo2.ul_lat)
    bottomY = max(geo1.lr_lat, geo2.lr_lat)

    if leftX < rightX and topY > bottomY:
        intersectionRect = GeoInformation()
        intersectionRect.ul_lon = leftX
        intersectionRect.ul_lat = topY
        intersectionRect.lr_lon = rightX
        intersectionRect.lr_lat = bottomY
        return intersectionRect
    return None
