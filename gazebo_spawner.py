import argparse
import rospy
import numpy as np
import copy

from gps.agent.utils.gazebo_spawner import GazeboModels
from gps.agent.utils.model import Model
from gps.agent.utils.basic_models import SPHERE, PEG_BOARD

rospy.init_node('gazebo_spawner_ur3e')

def place_target():
    sphere = SPHERE % ("target", "0.025", "GreenTransparent")
    model_names = ["target"]
    objpose = [[0.0131,  0.4019,  0.3026]]
    objpose = [[-0.13101034,  0.37818616,  0.50852045]]

    models = [[Model(model_names[0], objpose[0], file_type='string', string_model=sphere, reference_frame="base_link")]]
    GazeboModels(models, 'ur3_gazebo')


def place_models():
    model_names = ["multi_peg_board"]
    model_names = ["simple_peg_board"]
    objpose = [[0.217947,  0.367654, 0.75], None]
    objpose = [[0.402628, 0.275193, 0.807442], [0, -0.3556907, 0, 0.9346037]]
    objpose = [[0.198303, 0.244189, 0.75], None]
    objpose = [[-0.45, -0.20, 0.86], [0, 0.1986693, 0, 0.9800666]]
    objpose = [[-0.381961, -0.250909, 0.816082], None] #experiments so far
    models = [[Model(model_names[0], objpose[0], orientation=objpose[1])]]
    GazeboModels(models, 'ur3_gazebo')


def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument('--place', action='store_true',
                        help='Place models')
    parser.add_argument('--target', action='store_true',
                        help='Place targets')

    args = parser.parse_args()

    if args.place:
        place_models()
    if args.target:
        place_target()


if __name__ == "__main__":
    main()
