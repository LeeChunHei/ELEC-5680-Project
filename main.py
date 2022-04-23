import argparse

from env import Humanoid

class LoadArgFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        with values as f:
            parser.parse_args(f.read().split(), namespace)

if __name__ == '__main__':
    #parse argument
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--arg_file', type=open, action=LoadArgFromFile, help='argment file path')
    arg_parser.add_argument('--motion_file', type=str, default='./motion_file/sfu_walking.txt', help='motion file path')
    arg_parser.add_argument('--draw', action=argparse.BooleanOptionalAction, default=False, help='render the environment')
    arg_parser.add_argument('--timestep', type=float, default=1/240, help='simulation time step')
    arg_parser.add_argument('--fall_contact_bodies', type=int, nargs="+")
    args = arg_parser.parse_args()
    
    print(args)

