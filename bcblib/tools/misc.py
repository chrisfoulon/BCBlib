from collections import defaultdict
import random
from tqdm import tqdm

import numpy as np
from tqdm import tqdm

from bcblib.tools.general_utils import open_json
from scipy.stats import kruskal


def create_balanced_split(info_dict_keys, info_dict, num_splits=5):
    cluster_dict = defaultdict(list)
    bilat_offset = 2
    for k in info_dict_keys:
        clu_name = info_dict[k]['lesion_cluster']
        if clu_name == 'outside_clusters' or clu_name == 'empty_prediction':
            cluster_dict[clu_name].append(k)
        else:
            cluster_dict[clu_name[bilat_offset:]].append(k)
    # global_volumes = [info_dict[k]['volume'] for k in info_dict]
    #     print(np.mean(global_volumes))
    #     print(np.std(global_volumes))
    split_dict = [defaultdict(list) for i in range(num_splits)]

    # def len_splits(split_dict):
    #     len_list = np.zeros((len(split_dict)))
    #     for i, s in enumerate(split_dict):
    #         acc = 0
    #         for clu in s:
    #             acc += len(s[clu])
    #         len_list[i] = acc
    #     return len_list

    # rand_split = lambda l: random.choice(l)
    #     print(len(global_volumes))
    #     print((len(info_dict) // len(split_dict)))

    left_over_clu_dict = {}
    acc = 0
    for clu in cluster_dict:
        stop_length = len(cluster_dict[clu]) - (len(cluster_dict[clu]) % len(split_dict))
        # num_per_split = len(cluster_dict[clu]) // len(split_dict)
        for i in range(stop_length):
            split_dict[i % len(split_dict)][clu].append(cluster_dict[clu][i])
        left_over_num = len(cluster_dict[clu]) % len(split_dict)
        if left_over_num > 0:
            acc += left_over_num
            left_over_clu_dict[clu] = cluster_dict[clu][-left_over_num:]
    #     print(acc)
    #     print(np.sum([len(left_over_clu_dict[clu]) for clu in left_over_clu_dict]))

    counter = 0
    for clu in left_over_clu_dict:
        for k in left_over_clu_dict[clu]:
            split_dict[counter % len(split_dict)][clu].append(k)
            counter += 1

    #     print(len_splits(split_dict))
    #     print(np.sum(len_splits(split_dict)))
    for clu in split_dict[0]:
        s = f'{clu:28} : '
        for split in split_dict:
            s += f'{len(split[clu]):4} '
    #         print(s)

    split_volumes = []
    for split in split_dict:
        split_vols = []
        for clu in split:
            split_vols += [info_dict[k]['volume'] for k in split[clu]]
        split_volumes.append(split_vols)
    mean_splits = [np.mean(split) for split in split_volumes]
    std_splits = [np.std(split) for split in split_volumes]
    st, pval = kruskal(*split_volumes)
    return split_dict, mean_splits, std_splits, st, pval


def permutation_balanced_splits(info_dict_keys, info_dict, num_permutations):
    """
    Creates a balanced split of the data by permuting the keys of the info_dict and selecting the best split
    Parameters
    ----------
    info_dict_keys: list
    info_dict: dict
    num_permutations: int

    Returns
    -------

    Example:
    best_splits = permutation_balanced_splits(info_dict_keys, info_dict, 50000)
    """
    best_mean_range = np.inf
    best_std_range = np.inf
    best_pvalue = 0
    best_st = 0
    best_splits = None

    for perm in tqdm(range(num_permutations)):
        random.shuffle(info_dict_keys)
        split_dict, mean_splits, std_splits, st, pval = create_balanced_split(info_dict_keys, info_dict)
        means_range = (np.max(mean_splits) - np.min(mean_splits))
        stds_range = (np.max(std_splits) - np.min(std_splits))
        if best_mean_range > means_range and best_std_range > stds_range and best_pvalue < pval:
            best_mean_range = means_range
            best_std_range = stds_range
            best_st = st
            best_pvalue = pval
            best_splits = split_dict
    #         print(f'Means: {mean_splits}, Stds: {std_splits}, Stat: {st}, P-Value:{pval}')
    # Just converting the defaultdicts to dicts (easier for display and saving)
    best_splits = [dict(split) for split in best_splits]
    print(f' BEST SPLITS ===> Means: {best_mean_range}, Stds: {best_std_range}, Stat: {best_st}, P-Value:{best_pvalue}')
    return best_splits


