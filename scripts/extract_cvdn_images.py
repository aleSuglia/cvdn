#!/usr/bin/env python

''' Script to precompute image features using a Caffe ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT
    and VFOV parameters. '''

import argparse
import csv
import json
import math
import os
import sys

import cv2
import numpy as np

csv.field_size_limit(sys.maxsize)

# Caffe and MatterSim need to be on the Python path
sys.path.insert(0, 'build')
import MatterSim

# Simulator image parameters
WIDTH = 640
HEIGHT = 480
VFOV = 60

dataset_files = {
    "train": "train.jsonl",
    "val_seen": "val_seen.jsonl",
    "val_unseen": "val_unseen.jsonl"
}


def discretize_viewpoint(heading, elevation):
    M_PI = 3.14159265358979323846264338327950288
    heading_count = 12
    elevation_increment = M_PI / 6.0

    heading_increment = M_PI * 2.0 / heading_count
    heading_step = round(heading / heading_increment)

    if heading_step == heading_count:
        heading_step = 0

    if elevation < -elevation_increment / 2.0:
        view_index = heading_step
    elif elevation > elevation_increment / 2.0:
        view_index = heading_step + 2 * heading_count
    else:
        view_index = heading_step + heading_count

    return view_index


def extract_images_from_dataset(dataset_path, simulator):
    """
    Reads the entire dataset and extracts the images associated to the current split

    :param dataset_path: JSONL file containing the CVDN dataset
    :return:
    """

    with open(dataset_path) as in_file:
        for line in in_file:
            example = json.loads(line.strip())

            navigation_images = []
            action_images = []
            scan_id = example["scan"]

            for nav_camera in example["nav_camera"]:
                message_list = nav_camera["message"]

                for message in message_list:
                    viewIndex = discretize_viewpoint(message["heading"], message["elevation"])

                    navigation_images.append(get_rgb_from_coordinates(
                        simulator,
                        scan_id,
                        message["pano"],
                        viewIndex
                    ))

            for turn_data in example["actions"]:
                for action in turn_data:
                    action_images.append(get_rgb_from_coordinates(
                        simulator,
                        scan_id,
                        action["viewpointId"],
                        action["viewIndex"]
                    ))

            yield navigation_images, action_images


def get_rgb_from_coordinates(simulator, scanId, viewpointId, targetIdx):
    state = None
    for ix in range(targetIdx + 1):
        if ix == 0:
            simulator.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            simulator.makeAction([0], [1.0], [1.0])
        else:
            simulator.makeAction([0], [1.0], [0])

        state = simulator.getState()[0]
        assert state.viewIndex == ix

    # Write image.
    return np.float32(state.rgb), scanId, viewpointId, targetIdx


def main(args):
    # Set up the simulator
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.initialize()

    for split_id, split_name in dataset_files.items():
        dataset_path = os.path.join(args.dataset_folder, split_name)

        for navigation_images, action_images in extract_images_from_dataset(dataset_path, sim):
            # todo: do something with them!
            for image, scanId, viewpointId, targetIdx in action_images:
                cv2.imwrite(os.path.join(args.output_folder, "%s-%s-%d.png".format(scanId, viewpointId, targetIdx)), image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_folder', type=str, required=True)
    parser.add_argument('-output_folder', type=str, required=True)
    args = parser.parse_args()

    main(args)
