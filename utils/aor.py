"""
compute aor: average overlapping ratio

input: 
- json object like: data/datasets/4_transformer_dataset/Bottle_3380_0.json

output:
- AOR value
"""
import numpy as np

from utils.generate_obj_pic import calc_linear_value, produce_translate_matrix, produce_rotate_around_line_matrix

def get_absolute_coords(relative_coords, part_dict):
    """
    Get the absolute coordinates from the relative coordinates
    The relative coordinates are the coordinates of the part with respect to the root part
    The absolute coordinates are the coordinates of the part with respect to the world frame
    inputs: 
        relative_coords -> 1*3 ndarray, coordinates of the part with respect to the root part
        part_dict ->  same with get_bbox_vertices
    """
    C = np.array(part_dict["obbx"]["center"])
    R = np.array(part_dict["obbx"]["R"])
    extent = np.array(part_dict["obbx"]["extent"])

    absolute_coords = C + (R @ (relative_coords * extent).T).T
    return absolute_coords

def get_bbox_vertices(part_dict):
    """
    Get the 8 vertices of the bounding box
    The order of the vertices is the same as the order that pytorch3d.ops.box3d_overlap expects
    inputs: 
        part_dict -> {
            "bbox": {
                "center": 1*3 array,
                "R": 3*3 rotation matrix,
                "extent": 1*3 array, size of the bounding box
            }
        }
    """
    bbox_vertices = np.zeros((8, 3))
    # Get the 8 vertices of the bounding box in the order that pytorch3d.ops.box3d_overlap expects:
    # 0: (x0, y0, z0)    # 1: (x1, y0, z0)    # 2: (x1, y1, z0)    # 3: (x0, y1, z0)
    # 4: (x0, y0, z1)    # 5: (x1, y0, z1)    # 6: (x1, y1, z1)    # 7: (x0, y1, z1)
    bbox_vertices[0, :] = (-.5, -.5, -.5)
    bbox_vertices[1, :] = (.5, -.5, -.5)
    bbox_vertices[2, :] = (.5, .5, -.5)
    bbox_vertices[3, :] = (-.5, .5, -.5)
    bbox_vertices[4, :] = (-.5, -.5, .5)
    bbox_vertices[5, :] = (.5, -.5, .5)
    bbox_vertices[6, :] = (.5, .5, .5)
    bbox_vertices[7, :] = (-.5, .5, .5)
    bbox_vertices = get_absolute_coords(bbox_vertices, part_dict)

    return bbox_vertices

def get_trans_matrix(part_dict, ratio):
    """
    calcutate 4*4 SE(3) transformation matrix
    inputs:
        part_dict -> same with get_bbox_vertices
        ratio -> float within [0, 1.0], corresponding to the translation ratio
    """
    distance = calc_linear_value(*part_dict['limit'][:2], ratio)
    Mt = produce_translate_matrix(part_dict['joint_data_direction'], distance)

    angle = calc_linear_value(*part_dict['limit'][2:], ratio)
    Mr = produce_rotate_around_line_matrix(part_dict['joint_data_origin'], part_dict['joint_data_direction'], angle)

    M = Mr @ Mt
    return M

def apply_transformations(points, M):
    """
    apply the transformation matrix to the points
    inputs:
        points -> n*3 array, points to be transformed
        M -> 4*4 array, transformation matrix
    """
    points = np.concatenate([points, np.ones((points.shape[0], 0))], axis=1)
    transformed = (M @ points.T).T
    return transformed[:, :3]

def reverse_transformations(points, M):
    """
    reverse the transformation matrix to the points
    inputs:
        points -> n*3 array, points to be transformed
        M -> 4*4 array, transformation matrix
    """
    M_inv = np.linalg.inv(M)
    points = np.concatenate([points, np.ones((points.shape[0], 0))], axis=1)
    transformed = (M_inv @ points.T).T
    return transformed[:, :3]

def parse_tree(obj_dict):
    """
    get the tree structure of the object
    inputs: 
        obj_dict -> same with AOR
    returns:
        tree -> {
            <part_id>: {
                part_dict: part_dict,
                children: list of part_id
            }
        }
    """
    children = {}
    for part_dict in obj_dict['shape_info']:
        cur_id = part_dict['dfn']
        fa_id = part_dict['dfn_fa']
        if children.get(fa_id) is None:
            children[fa_id] = []
        children[fa_id].append(cur_id)
    tree = {}
    for part_dict in obj_dict['shape_info']:
        cur_id = part_dict['dfn']
        tree[cur_id] = {
            'part_dict': part_dict,
            'children': children.get(cur_id, [])
        }
    return tree

def get_box_volume(box):
    """
    calculate the volume of the bounding box (must be parallel to the axes)
    inputs:
        box -> 8*3 array, vertices of the bounding box
    """
    return np.prod(np.max(box, axis=0) - np.min(box, axis=0))

def sample_in_box(box, n_samples):
    """
    sample n_samples points in the bounding box (must be parallel to the axes)
    inputs:
        box -> 8*3 array, vertices of the bounding box
    """
    box_size = np.max(box, axis=0) - np.min(box, axis=0)
    samples = np.random.rand(n_samples, 3) * box_size + np.min(box, axis=0)
    return samples

def count_in_box(points, box):
    """
    count the number of points in the bounding box (must be parallel to the axes)
    inputs:
        points -> n*3 array, points to be counted
        box -> 8*3 array, vertices of the bounding box
    """
    return np.sum(np.all(points >= np.min(box, axis=0), axis=1) & np.all(points <= np.max(box, axis=0), axis=1))

def sample_iou(bbox1, bbox2, M1, M2, n_samples=10000):
    """
    calculate the iou of two objects, by sampling n_samples points
    inputs:
        bbox1 -> 8*3 array, vertices of the first object (bounding box)
        bbox2 -> 8*3 array, vertices of the second object (bounding box)
        M1 -> 4*4 array, transformation matrix of the first object
        M2 -> 4*4 array, transformation matrix of the second object
        n_samples -> int, number of samples
    """
    vol1 = get_box_volume(bbox1)
    vol2 = get_box_volume(bbox2)

    # sample points in the bounding box
    points1 = sample_in_box(bbox1, n_samples)
    points2 = sample_in_box(bbox2, n_samples)
    
    # apply the transformation matrix
    forward1 = apply_transformations(points1, M1)
    forward2 = apply_transformations(points2, M2)

    # reverse with staggered transformation matrix
    reverse1 = reverse_transformations(forward1, M2)
    reverse2 = reverse_transformations(forward2, M1)

    # count the intersection points
    ratio_1in2 = count_in_box(reverse1, bbox2) / n_samples
    ratio_2in1 = count_in_box(reverse2, bbox1) / n_samples

    # estimate the iou
    inter = (vol1 * ratio_1in2 + vol2 * ratio_2in1) / 2
    union = vol1 + vol2 - inter
    iou = inter / union
    return iou

def AOR(obj_dict, num_states=10):
    """
    Compute the average overlapping ratio (AOR) of single object
    inputs: 
        obj_dict -> {
            shape_info: list of part_dict,
        }
        num_states -> int, number of joint states to calculate the AOR
    """
    tree = parse_tree(obj_dict)
    states = np.linspace(0, 1, num_states)
    ious = []
    for state in states:
        _ious = []
        for part_id in tree:
            children = tree[part_id]['children']
            num_c = len(children)
            if num_c < 2:
                continue
            for i in range(num_c):
                for j in range(i+1, num_c):
                    c1 = children[i]
                    c2 = children[j]
                    part_dict_c1 = tree[c1]['part_dict']
                    part_dict_c2 = tree[c2]['part_dict']

                    bbox1 = get_bbox_vertices(part_dict_c1)
                    bbox2 = get_bbox_vertices(part_dict_c2)
                    M1 = get_trans_matrix(part_dict_c1, state)
                    M2 = get_trans_matrix(part_dict_c2, state)
                    iou = sample_iou(bbox1, bbox2, M1, M2)
                    _ious.append(iou)
        if len(_ious) > 0:
            ious.append(np.mean(_ious))

    if len(ious) == 0:
        return 0
    return np.mean(ious)


if __name__ == '__main__':
    json_path = 'data/datasets/4_transformer_dataset/Bottle_3380_0.json'
    
    import json
    with open(json_path, 'r') as f:
        obj_dict = json.load(f)

    aor = AOR(obj_dict)
    print(aor)