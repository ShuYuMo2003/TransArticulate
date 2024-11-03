import numpy as np
import open3d as o3d
from tqdm import tqdm
from rich import print

import sys
sys.path.append('..')

from utils.generate_obj_pic import calc_linear_value, produce_translate_matrix, produce_rotate_around_line_matrix

def get_trans_matrix(part_dict, ratio):
    """
    calcutate 4*4 SE(3) transformation matrix
    ratio: float within [0, 1.0], corresponding to the translation ratio
    """
    distance = calc_linear_value(*part_dict['limit'][:2], ratio)
    Mt = produce_translate_matrix(part_dict['joint_data_direction'], distance)

    angle = calc_linear_value(*part_dict['limit'][2:], ratio)
    Mr = produce_rotate_around_line_matrix(part_dict['joint_data_origin'], part_dict['joint_data_direction'], angle)

    M = Mr @ Mt
    return M

def prepare_trans_matrix(obj, ratio):
    """
    prepare the transformation matrix for every part in the object
    """

    M_dict, fa = {}, {}
    for part in obj:
        cur_id = part['dfn']
        print('cur_id = ', cur_id)
        M = get_trans_matrix(part, ratio)
        M_dict[cur_id] = M
        fa[cur_id] = part['dfn_fa']

    keys = list(M_dict.keys())
    keys.sort()
    for cur_id in keys:
        if fa[cur_id] >= 0 and fa[cur_id] != cur_id:
            M_dict[cur_id] = M_dict[cur_id] @ M_dict[fa[cur_id]]
    return M_dict

def apply_transformations(points, M):
    """
    apply the transformation matrix to the points
    points: n*3 array, points to be transformed
    M: 4*4 array, transformation matrix
    """
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    transformed = (M @ points.T).T
    return transformed[:, :3]

def reverse_transformations(points, M):
    """
    reverse the transformation matrix to the points
    points: n*3 array, points to be transformed
    M: 4*4 array, transformation matrix
    """
    M_inv = np.linalg.inv(M)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    transformed = (M_inv @ points.T).T
    return transformed[:, :3]

def get_min_distance(query_point, points, n_sample=None):
    """
    get the minimum distance between a query point and a set of points
    """
    if n_sample is not None:
        n_sample = min(n_sample, points.shape[0])
        index = np.random.choice(points.shape[0], n_sample, replace=False)
        points = points[index]
    distances = np.linalg.norm(points - query_point, axis=1)
    return np.min(distances)

def get_ref_seperation(points, ref_sample=5, n_sample=None):
    """
    assume points to be uniformly distributed, estimate the seperation distance
    """
    ref_sample = min(ref_sample, points.shape[0] // 2)

    ref_index = np.random.choice(points.shape[0], ref_sample, replace=False)
    ref_points = points[ref_index]
    other_points = np.delete(points, ref_index, axis=0)

    min_distances = []
    for ref_point in ref_points:
        min_distance = get_min_distance(ref_point, other_points, n_sample)
        min_distances.append(min_distance)
    return np.mean(min_distances)

def count_intersected_points(query_points, ref_points, sep, n_sample=None):
    """
    count the number of intersected points between query points and reference points
    """
    sample_ratio = 1
    if n_sample is not None:
        n_sample = min(n_sample, query_points.shape[0])
        sample_ratio = n_sample / query_points.shape[0]
        index = np.random.choice(query_points.shape[0], n_sample, replace=False)
        query_points = query_points[index]

    n_intersected = 0
    for query_point in query_points:
        min_distance = get_min_distance(query_point, ref_points, n_sample)
        if min_distance < sep:
            n_intersected += 1
    return n_intersected / sample_ratio

def sample_iou(points1, points2, M1, M2, conf_T, n_sample=None):
    """
    compute iou between two parts, after applying a SE(3) transformation
    """
    # apply the transformation
    trans1 = apply_transformations(points1, M1)
    trans2 = apply_transformations(points2, M2)

    print('points1.shape = ', points1.shape)
    print('points2.shape = ', points2.shape)

    # get the seperation distance
    sep1 = get_ref_seperation(trans1, n_sample=n_sample)
    print('sep1 = ', sep1)
    sep2 = get_ref_seperation(trans2, n_sample=n_sample)
    print('sep2 = ', sep2)
    # sep1 = 2 / 256
    # sep2 = 2 / 256

    # compute iou
    inter1 = count_intersected_points(trans1, trans2, sep2 * conf_T, n_sample)
    print('inter1 = ', inter1)
    inter2 = count_intersected_points(trans2, trans1, sep1 * conf_T, n_sample)
    print('inter2 = ', inter2)
    inter = (inter1 + inter2) / 2
    union = trans1.shape[0] + trans2.shape[0] - inter
    iou = inter / union
    return iou

def POR(obj, n_sample=10, n_states=10, conf_T=1.5):
    print("n_sample = ", n_sample)
    """
    compute the average and maximum Part Overlapping Ratio (POR) of a object
    n_states: number of poses for the object
    conf_T: confidence threshold, the ratio of the seperation distance in [1, 2]
        bigger the value, more false positive (FP)
        smaller the value, more false negative (FN)
    n_sample: if this fucntion is too slow, you can set n_sample to a smaller value (None for no sampling)
    returns: (average POR, maximum POR)
    """
    assert type(obj) == list, """
obj must be a list of parts with the following structure:
[
    {
        "points": n*4 np.ndarray,
        "joint_data_origin": [x0, y0, z0],
        "joint_data_direction": [x1, y1, z1],
        "limit": [p_min, p_max, r_min, r_max],
        "dfn": dfs number,
        "dfn_fa": father's dfs number
    }, ... (other parts)
]
"""

    if n_sample is not None:
        print("sampling points")
        for part in obj:
            index = np.random.choice(part['points'].shape[0], n_sample, replace=False)
            part['points'] = part['points'][index]

    n_parts = len(obj)
    states = np.linspace(0, 1, n_states)
    results = []
    for state in tqdm(states, desc="Processing on different pose state."):
        M_dict = prepare_trans_matrix(obj, state)
        ious = []
        for i in range(n_parts):
            for j in range(i + 1, n_parts):
                points1 = obj[i]['points'][:, :3]
                points2 = obj[j]['points'][:, :3]
                M1 = M_dict[obj[i]['dfn']]
                M2 = M_dict[obj[j]['dfn']]
                iou = sample_iou(points1, points2, M1, M2, conf_T, n_sample)
                if iou is not None:
                    ious.append(iou)
                # if i == 0 and j == 2:
                #     print(f"State: {state}, Part {i} and Part {j}: {iou}")
        if len(ious) > 0:
            results.append(np.mean(ious))

    if len(results) == 0:
        return None, None
    return np.mean(results), np.max(results)

if __name__ == "__main__":
    # example usage
    # obj = []
    # n_parts = 3
    # for i in range(n_parts):
    #     points = np.random.rand(1000, 4)
    #     joint_data_origin = [0.5, 0.5, 0.5]
    #     joint_data_direction = [1, 0, 0]
    #     if i == 0:
    #         limit = [0, 0, 0, 0]
    #     elif i == 1:
    #         limit = [0, 1, 0, 0]
    #     else:
    #         limit = [0, 0, 0, 2*np.pi]
    #     obj.append({
    #         "points": points,
    #         "joint_data_origin": joint_data_origin,
    #         "joint_data_direction": joint_data_direction,
    #         "limit": limit,
    #         "dfn": i,
    #         "dfn_fa": [-1, 0, 1][i]
    #     })

    import pickle
    obj_list = pickle.load(open("/root/workspace/crc61cnhri0c7384uggg/TransArticulate/log/TF-Diff/10-09-11PM-05-47/output-0.data", "rb"))
    obj = obj_list[0]['data']

    avg_por, max_por = POR(obj, 4096, n_states=10)
    print(f"Average POR: {avg_por}")
    print(f"Maximum POR: {max_por}")