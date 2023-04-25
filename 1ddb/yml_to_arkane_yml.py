#!/usr/bin/env python3
# encoding: utf-8

import argparse

from arc.common import read_yaml_file, save_yaml_file


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.
    This uses the :mod:`argparse` module, which ensures that the command-line arguments are
    sensible, parses them, and returns them.
    """
    parser = argparse.ArgumentParser(description='yaml to arkane yaml format converter')
    parser.add_argument('--yml-path', metavar='input', type=str,
                        help='A path to the YAML input file, including the name')
    parser.add_argument('--out-path', metavar='output', type=str,
                        help='A path to the desired output file, including the name')
    args = parser.parse_args(command_line_args)
    return args

def main():
    args = parse_command_line_arguments()
    try:
        _input = read_yaml_file(str(args.yml_path))
    except FileNotFoundError:
        print("file not found")
        return
    save_yaml_file(path = str(args.out_path),
                   content= {"angle_unit" : "degrees",
                             "energy_unit": "kJ/mol",
                             "angles"     : list(map(lambda x: float(x[0]) ,_input["directed_scan"].keys())),
                             "energies"   : [float(_input["directed_scan"][instance]["energy"]) for instance in _input["directed_scan"]],
                            },
                   )

if __name__ == "__main__":
    main()
