import argparse
import json
from pathlib import Path

from preprocessing.datasets_utils import process_all_tasks


def get_parser():
    parser_ = argparse.ArgumentParser(description="")
    parser_.add_argument("--params_file", type=str, help="Path to file with params",
                         default="preprocessing_params.json")
    parser_.add_argument("--verbose", default=False, action='store_true')
    return parser_


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    params_file = args.params_file
    with open(params_file) as json_file:
        data = json.load(json_file)
    norm_ct = data.get("normalize_ct")
    norm_volume = data.get("normalize_volume")
    output_format = data.get("output_format")
    process_all_tasks(root=Path.cwd(), tasks_list=data.get("tasks"),
                      norm_ct=norm_ct, normalize_volume=norm_volume, output_format=output_format)
