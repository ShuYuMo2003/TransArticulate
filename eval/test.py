import numpy as np
from renders import get_bbox_mesh_pair


def test_get_bbox_mesh_pair():
    mesh = get_bbox_mesh_pair(np.array([0, 0, 0]), np.array([1, 2, 3]))
    mesh.export('bbx.obj')

if __name__ == '__main__':
    test_get_bbox_mesh_pair()