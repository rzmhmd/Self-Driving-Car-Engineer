# ---------------------------------------------------------------------
# Exercises from lesson 1 (lidar)
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Starter Code
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2
from PIL import Image
import io
import sys
import os
import cv2
import numpy as np
import zlib

# Add current working directory to path
sys.path.append(os.getcwd())

# Waymo open dataset reader


# Exercise C1-5-5 : Visualize intensity channel
def vis_intensity_channel(frame, lidar_name):

    print("Exercise C1-5-5")
    # extract range image from frame
    lidar = [obj for obj in frame.lasers if obj.name ==
             lidar_name][0]  # get laser data structure from frame
    ri = []
    if len(lidar.ri_return1.range_image_compressed) > 0:  # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(
            lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
    ri[ri < 0] = 0.0
    # map value range to 8bit
    ri_intensity = ri[:, :, 1]
    ri_intensity = (np.amax(ri_intensity)/2) * ri_intensity * \
        255 / (np.amax(ri_intensity) - np.amin(ri_intensity))
    img_intensity = ri_intensity.astype(np.uint8)

    # focus on +/- 45° around the image center
    deg45 = int(img_intensity.shape[1] / 8)
    ri_center = int(img_intensity.shape[1]/2)
    img_intensity = img_intensity[:, ri_center-deg45:ri_center+deg45]

    cv2.imshow('intensity image', img_intensity)
    cv2.waitKey(0)


# Exercise C1-5-2 : Compute pitch angle resolution
def print_pitch_resolution(frame, lidar_name):

    print("Exercise C1-5-2")
    # load range image
    lidar = [obj for obj in frame.lasers if obj.name ==
             lidar_name][0]  # get laser data structure from frame
    ri = []
    if len(lidar.ri_return1.range_image_compressed) > 0:  # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(
            lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)

    # compute vertical field-of-view from lidar calibration
    # get laser calibration
    lidar_calib = [
        sensor for sensor in frame.context.laser_calibrations if sensor.name == lidar_name][0]
    min_pitch = lidar_calib.beam_inclination_min
    max_pitch = lidar_calib.beam_inclination_max
    VFOV = max_pitch - min_pitch

    # compute pitch resolution and convert it to angular minutes
    Pitch_Res_R = VFOV / ri.shape[0]
    Pitch_Res_D = 60 * Pitch_Res_R * 180 / np.pi
    print("pitch angle resolution = " + '{0:.2f}'.format(Pitch_Res_D) + "°")


# Exercise C1-3-1 : print no. of vehicles
def print_no_of_vehicles(frame):

    num_vehicles = 0
    for label in frame.laser_labels:
        if label.type == label.TYPE_VEHICLE:
            num_vehicles += num_vehicles

    print("number of labeled vehicles in current frame = " + str(num_vehicles))
