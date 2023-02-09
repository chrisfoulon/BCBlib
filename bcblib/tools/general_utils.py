import json

import numpy as np


def open_json(path):
    with open(path, 'r') as j:
        return json.load(j)


def save_json(path, d):
    with open(path, 'w+') as j:
        return json.dump(d, j, indent=4)


def save_list(path, li):
    np.savetxt(path, li, delimiter='\n', fmt='%s')