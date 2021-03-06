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

def place_soft():
    name = "simple_peg_board"
    objpose = [[-0.41, -0.25, 0.816082], [0, 0.0344632, 0, 0.999406]]
    string_model = PEG_BOARD.format(1e6)
    models = [[Model(name, objpose[0], orientation=objpose[1], file_type='string', string_model=string_model, reference_frame="world")]]
    GazeboModels(models, 'ur3_gazebo')

def place_door():
    name = "hinged_door"
    objpose = [[-0.40, 0.20, 0.76], [ 0, 0, 0.9238795, 0.3826834 ]]
    models = [[Model(name, objpose[0], orientation=objpose[1])]]
    GazeboModels(models, 'ur3_gazebo')

def place_ring():
    name = "lens_ring"
    objpose = [[-0.370381, -0.23, 0.82], [ 0, 0, -0.7071068, 0.7071068 ]]
    models = [[Model(name, objpose[0], orientation=objpose[1])]]
    GazeboModels(models, 'ur3_gazebo')

def place_groove():
    name = "groove"
    objpose = [[-0.11827383, -0.449, 0.36202397], [-0.5, -0.5, -0.5, 0.5]]
    models = [[Model(name, objpose[0], orientation=objpose[1], reference_frame="base_link")]]
    GazeboModels(models, 'ur3_gazebo')

def place_groovemiddle():
    name = "groove_middle"
    objpose = [[-0.11827383, -0.449, 0.36202397], [-0.5, -0.5, -0.5, 0.5]]
    models = [[Model(name, objpose[0], orientation=objpose[1], reference_frame="base_link")]]
    GazeboModels(models, 'ur3_gazebo')

def place_groovebig():
    name = "groove_big"
    # objpose = [[-0.10027383, -0.48, 0.48202397], [-0.5,-0.5,-0.5,0.5 ]]
    objpose = [[-0.11827383, -0.449, 0.36202397], [-0.5, -0.5, -0.5, 0.5]]
    models = [[Model(name, objpose[0], orientation=objpose[1], reference_frame="base_link")]]
    GazeboModels(models, 'ur3_gazebo')

def place_mortisebig():
    name = "mortise_big"
    objpose = [[-0.11827383+0.0825, -0.449-0.013+0.005, 0.36202397], [-0.5, -0.5, -0.5, 0.5]]
    models = [[Model(name, objpose[0], orientation=objpose[1], reference_frame="base_link")]]
    GazeboModels(models, 'ur3_gazebo')

def place_mortisex():
    name = "mortise_x"
    objpose = [[-0.11827383+0.08, -0.449-0.013-0.01, 0.36202397], [-0.5, -0.5, -0.5, 0.5]]
    models = [[Model(name, objpose[0], orientation=objpose[1], reference_frame="base_link")]]
    GazeboModels(models, 'ur3_gazebo')

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument('--place', action='store_true',
                        help='Place models')
    parser.add_argument('--target', action='store_true',
                        help='Place targets')
    parser.add_argument('--soft', action='store_true',
                        help='Place soft peg board')
    parser.add_argument('--door', action='store_true',
                        help='Place door')
    parser.add_argument('--ring', action='store_true',
                        help='Place ring')
    parser.add_argument('--groove', action='store_true',
                        help='Place groove')
    parser.add_argument('--groove_middle', action='store_true',
                        help='Place groove_middle')
    parser.add_argument('--groove_big', action='store_true',
                        help='Place groove_big')
    parser.add_argument('--mortise_big', action='store_true',
                        help='Place mortise_big')
    parser.add_argument('--mortise_x', action='store_true',
                        help='Place mortise_x')
    args = parser.parse_args()

    if args.place:
        place_models()
    if args.target:
        place_target()
    if args.soft:
        place_soft()
    if args.door:
        place_door()
    if args.ring:
        place_ring()
    if args.groove:
        place_groove()
    if args.groove_middle:
        place_groovemiddle()
    if args.groove_big:
        place_groovebig()
    if args.mortise_big:
        place_mortisebig()
    if args.mortise_x:
        place_mortisex()
if __name__ == "__main__":
    main()
