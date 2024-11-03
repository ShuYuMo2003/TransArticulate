import numpy as np
import open3d as o3d

def get_value_from_ratio(ratio, min_value, max_value):
    return min_value + ratio * (max_value - min_value)

def get_translation_tensor(part, R):
    """
    compute the translation tensor of a part, given a ratio R
    returns: 1x3 np.array
    """
    direction = np.array(part["joint_data_direction"])
    direction = direction / np.linalg.norm(direction)
    distance = get_value_from_ratio(R, part["limit"][0], part["limit"][1])
    translation = direction * distance
    # return o3d.core.Tensor(translation)
    return translation

def get_rotation_tensor(part, R):
    """
    compute the rotation tensor of a part, given a ratio R
    returns: 3x3 np.array
    """
    axis = np.array(part["joint_data_direction"])
    axis = axis / np.linalg.norm(axis)
    
    angle = get_value_from_ratio(R, part["limit"][2], part["limit"][3])
    # angle = np.radians(angle)

    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    ux, uy, uz = axis

    rotation = np.array([
        [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
        [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
        [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
    ])
    # return o3d.core.Tensor(rotation)
    return rotation

# def get_wtmesh(mesh):
#     """
#     convert mesh to watertight mesh, if not already watertight
#     inputs: mesh->open3d.geometry.TriangleMesh
#     returns: open3d.geometry.TriangleMesh
#     """
#     if not mesh.is_watertight():
#         mesh, _ = mesh.compute_convex_hull()
#     return mesh

# def get_wtmesh(mesh):
#     """
#     convert mesh to watertight mesh, if not already watertight
#     inputs: mesh->open3d.geometry.TriangleMesh
#     returns: open3d.geometry.TriangleMesh
#     """
#     if not mesh.is_watertight():
#         mesh.remove_duplicated_vertices()
#         mesh.remove_duplicated_triangles()
#     assert mesh.is_watertight() == True, \
#         "failed to generate watertight mesh"
#     return mesh


def transform_iou(part1, part2, trans_R):
    """
    compute iou between two parts, after applying a SE(3) transformation
    trans_R: overall ratio of the transformation
    """
    mesh1 = part1["mesh"]
    mesh2 = part2["mesh"]
    assert type(mesh1) == o3d.geometry.TriangleMesh and type(mesh2) == o3d.geometry.TriangleMesh, \
        "part.mesh must be an open3d.geometry.TriangleMesh object"
    assert not mesh1.is_empty() and len(mesh1.vertices) > 0 and len(mesh1.triangles) > 0 \
        and not mesh2.is_empty() and len(mesh2.vertices) > 0 and len(mesh2.triangles) > 0, \
        "part.mesh must not be empty"
    
    # transform part1
    T1 = get_translation_tensor(part1, trans_R)
    R1 = get_rotation_tensor(part1, trans_R)
    center1 = np.array(part1["joint_data_origin"])
    mesh1.translate(T1)
    mesh1.rotate(R1, center1)

    # transform part2
    T2 = get_translation_tensor(part2, trans_R)
    R2 = get_rotation_tensor(part2, trans_R)
    center2 = np.array(part2["joint_data_origin"])
    mesh2.translate(T2)
    mesh2.rotate(R2, center2)

    # prepare for boolean operations
    mesh1.orient_triangles()
    mesh2.orient_triangles()
    mesh1 = o3d.t.geometry.TriangleMesh.from_legacy(mesh1)
    mesh2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh2)
    
    # compute iou
    # union = mesh1.boolean_union(mesh2).to_legacy()
    # intersection = mesh1.boolean_intersection(mesh2).to_legacy()
    # union, intersection = get_wtmesh(union), get_wtmesh(intersection)
    # o3d.io.write_triangle_mesh(f"u_{trans_R}.ply", union)
    # o3d.io.write_triangle_mesh(f"i_{trans_R}.ply", intersection)
    # iou = intersection.get_volume() / union.get_volume()
    # return iou
    try:
        union = mesh1.boolean_union(mesh2).to_legacy()
        intersection = mesh1.boolean_intersection(mesh2).to_legacy()
        union, intersection = get_wtmesh(union), get_wtmesh(intersection)
        o3d.io.write_triangle_mesh(f"u_{trans_R}.ply", union)
        o3d.io.write_triangle_mesh(f"i_{trans_R}.ply", intersection)
        iou = intersection.get_volume() / union.get_volume()
        return iou
    except:
        return 0


def POR(obj, n_states=10):
    """
    compute the average and maximum Part Overlapping Ratio (POR) of a object
    returns: (average POR, maximum POR)
    """
    assert type(obj) == list, """
obj must be a list of parts with the following structure:
[
    {
        "mesh": open3d.geometry.TriangleMesh,
        "joint_data_origin": [x0, y0, z0],
        "joint_data_direction": [x1, y1, z1],
        "limit": [p_min, p_max, r_min, r_max],
    }, ... (other parts)
]
"""
    n_parts = len(obj)
    states = np.linspace(0, 1, n_states)
    print(f"Computing POR for {n_parts} parts with {states}")
    
    results = []
    for state in states:
        ious = []
        for i in range(n_parts):
            for j in range(i + 1, n_parts):
                iou = transform_iou(obj[i], obj[j], state)
                if iou is not None:
                    ious.append(iou)
        if len(ious) > 0:
            results.append(np.mean(ious))

    if len(results) == 0:
        return None, None
    return np.mean(results), np.max(results)

if __name__ == "__main__":
    # example usage
    obj = []
    n_parts = 2
    for i in range(n_parts):
        mesh = o3d.geometry.TriangleMesh.create_cone()
        joint_data_origin = [0, 0, 0]
        joint_data_direction = [1, 0, 0]
        if i == 0:
            limit = [0, 0, 0, 0]
        elif i == 1:
            limit = [0, 1, 0, 0]
        else:
            limit = [0, 0, 0, 2*np.pi]
        obj.append({
            "mesh": mesh,
            "joint_data_origin": joint_data_origin,
            "joint_data_direction": joint_data_direction,
            "limit": limit
        })
    avg_por, max_por = POR(obj, n_states=2)
    print(f"Average POR: {avg_por}")
    print(f"Maximum POR: {max_por}")

    # mesh1 = o3d.geometry.TriangleMesh.create_cone()
    # mesh2 = o3d.geometry.TriangleMesh.create_cone()
    # mesh2 = mesh2.translate([1.99, 0, 0])
    # mesh1 = o3d.t.geometry.TriangleMesh.from_legacy(mesh1)
    # mesh2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh2)
    # union = mesh1.boolean_union(mesh2)
    # union = union.to_legacy()
    # o3d.io.write_triangle_mesh("union.ply", union)

    # mesh = o3d.geometry.TriangleMesh.create_cone()
    # print(mesh.get_volume())