import json
import math
import os
import sys
from argparse import ArgumentParser

sys.path.append("build")
import MatterSim
import networkx as nx

from utils import load_nav_graphs

parser = ArgumentParser()
parser.add_argument("-output", help="Path to the refined dataset folder")
parser.add_argument("-dataset_path", help="Path to the dataset folder")

split_files = {
    "train": "train.jsonl",
    "valid_seen": "val_seen.jsonl",
    "valid_unseen": "val_unseen.jsonl"
}

EPISODE_LENGTH = 20


def load_nav_graphs_paths(scans):
    graphs = load_nav_graphs(scans)
    paths = {}
    for scan, G in graphs.iteritems():  # compute all shortest paths
        paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
    distances = {}
    for scan, G in graphs.iteritems():  # compute all shortest paths
        distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    return graphs, paths, distances


def shortest_path_action(state, goalViewpointId, paths, graphs):
    # 'left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>'
    ''' Determine next action on the shortest path to goal, for supervised training. '''
    if state.location.viewpointId == goalViewpointId:
        return {"command": (0, 0, 0), "action": "<ignore>"}  # do nothing
    path = paths[state.scanId][state.location.viewpointId][goalViewpointId]
    nextViewpointId = path[1]
    # Can we see the next viewpoint?
    for i, loc in enumerate(state.navigableLocations):
        if loc.viewpointId == nextViewpointId:
            # Look directly at the viewpoint before moving
            if loc.rel_heading > math.pi / 6.0:
                return {"command": (0, 1, 0), "action": "right"}  # Turn right
            elif loc.rel_heading < -math.pi / 6.0:
                return {"command": (0, -1, 0), "action": "left"}  # Turn left
            elif loc.rel_elevation > math.pi / 6.0 and state.viewIndex // 12 < 2:
                return {"command": (0, 0, 1), "action": "up"}  # Look up
            elif loc.rel_elevation < -math.pi / 6.0 and state.viewIndex // 12 > 0:
                return {"command": (0, 0, -1), "action": "down"}  # Look down
            else:
                return {"command": (i, 0, 0), "action": "forward"}  # Move

    # Can't see it - first neutralize camera elevation
    if state.viewIndex // 12 == 0:
        return {"command": (0, 0, 1), "action": "up"}  # Look up
    elif state.viewIndex // 12 == 2:
        return {"command": (0, 0, -1), "action": "down"}  # Look down
    # Otherwise decide which way to turn
    pos = [state.location.x, state.location.y, state.location.z]
    target_rel = graphs[state.scanId].node[nextViewpointId]['position'] - pos
    target_heading = math.pi / 2.0 - math.atan2(target_rel[1], target_rel[0])  # convert to rel to y axis
    if target_heading < 0:
        target_heading += 2.0 * math.pi
    if state.heading > target_heading and state.heading - target_heading < math.pi:
        return {"command": (0, -1, 0), "action": "left"}  # Turn left
    if target_heading > state.heading and target_heading - state.heading > math.pi:
        return {"command": (0, -1, 0), "action": "left"}  # Turn left
    return {"command": (0, 1, 0), "action": "right"}  # Turn right


# it works by reference
def generate_actions_for_split(dataset, simulator):
    scans = [item["scan"] for item in dataset]
    graphs, paths, distances = load_nav_graphs_paths(scans)

    # we extract from the nav camera the information associated to the current path
    for item in dataset:
        scanIds = [item["scan"]]
        viewpointIds = [item["planner_path"][0]]
        headings = [item['start_pano']['heading']]

        actions_metadata = []
        state = None

        simulator.newEpisode(scanIds, viewpointIds, headings, [0])

        for camera in item["nav_camera"]:
            turn_actions = []

            for message in camera["message"]:
                for i, state in enumerate(simulator.getState()):
                    action_metadata = shortest_path_action(state, message["pano"], paths, graphs)
                    command = action_metadata["command"]
                    i, h, e = int(command[0]), float(command[1]), float(command[2])

                    simulator.makeAction([i], [h], [e])

                    turn_actions.append(
                        {
                            'viewpoint': state.location.viewpointId,
                            'viewIndex': state.viewIndex,
                            'heading': state.heading,
                            'elevation': state.elevation,
                            'step': state.step,
                            'action': action_metadata
                        }
                    )

            actions_metadata.append(turn_actions)

        item["actions"] = actions_metadata


def main(args):
    print("Initialising the Matterport simulator")
    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.setCameraResolution(640, 480)
    sim.setCameraVFOV(math.radians(60))
    sim.initialize()

    for split, split_file in split_files.items():
        full_path = os.path.join(args.dataset_path, split_file)
        print("Processing split: {}".format(full_path))
        with open(full_path) as in_file:
            dataset = [json.loads(line.strip()) for line in in_file]

        generate_actions_for_split(dataset, sim)

        full_mod_path = os.path.join(args.output, split_file)
        print("Writing updated version of the dataset file: {}".format(full_mod_path))
        with open(full_mod_path, mode="w") as out_file:
            out_file.writelines(json.dumps(ex) for ex in dataset)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
