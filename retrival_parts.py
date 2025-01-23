import pickle
import json
import time
import trimesh
import point_cloud_utils as pcu
from rich import print
import numpy as np
from pathlib import Path
from tqdm import tqdm

src_path = Path('data/datasets/6_ours_obj_dats')
dst_path = Path('data/datasets/6_ours_re_obj_dats')
inf_path = Path('data/datasets/1_preprocessed_info')
mes_path = Path('data/datasets/1_preprocessed_mesh')
tmp_path = Path('temp')

num_sample_points = 1000000

import shutil
shutil.rmtree(dst_path, ignore_errors=True)
shutil.rmtree(tmp_path, ignore_errors=True)
dst_path.mkdir(exist_ok=True, parents=True)
tmp_path.mkdir(exist_ok=True, parents=True)
failed = []

def mesh_to_point(mesh_path: Path, mesh_name: str, cur_bbx):
    s_time = time.time()
    print("Generate sampling point for ", mesh_path)

    mesh_obj_path = tmp_path / (mesh_name + '.obj')

    mesh = trimesh.load_mesh(mesh_path)
    (min_bound, max_bound) = mesh.bounds
    center, lxyz = cur_bbx[0, :], cur_bbx[1, :]
    tg_min_bound = center - lxyz / 2
    tg_max_bound = center + lxyz / 2
    print(tg_max_bound - max_bound)
    print(tg_min_bound - min_bound)

    mesh.vertices = tg_min_bound + (tg_max_bound - tg_min_bound) * (
        (mesh.vertices - min_bound) / (max_bound - min_bound)
    )
    min_bound, max_bound = tg_min_bound, tg_max_bound
    mesh.export(mesh_obj_path)

    v, f = pcu.load_mesh_vf(mesh_obj_path)
    resolution = 12_000
    vw, fw = pcu.make_mesh_watertight(v, f, resolution)

    points = np.random.uniform(low=min_bound, high=max_bound, size=(num_sample_points, 3))
    sdf, fid, bc = pcu.signed_distance_to_mesh(points, vw, fw)
    inside_point = points[sdf < 0]

    rho = num_sample_points / (max_bound - min_bound).prod()

    print("rho =", rho, "inside_point shape =", inside_point.shape[0], "time used =", time.time() - s_time, " inside_ratio =", inside_point.shape[0] / num_sample_points)

    return inside_point, rho

for src_dat_path in tqdm(list(src_path.glob('*.dat')), 'QAQ 0'):
    dat = pickle.loads(open(src_dat_path, 'rb').read())
    dat_info = src_dat_path.stem.split('_')
    shape_name = '_'.join(dat_info[:2])

    exist_bbx = []
    if not (inf_path / (shape_name + '.json')).exists():
        failed.append(src_dat_path)
        with open(dst_path / (src_dat_path.stem + '.dat'), 'wb') as f:
            f.write(pickle.dumps(dat))
        continue

    json_content = json.loads((inf_path / (shape_name + '.json')).read_text())
    for part in tqdm(json_content['part']):
        bbx = part['bbx']
        lxyz = (np.array(bbx[1]) - np.array(bbx[0]))
        center = (np.array(bbx[1]) + np.array(bbx[0])) / 2
        exist_bbx.append((np.array([center, lxyz]), part['mesh']))

    for idx, part in enumerate(dat):
        cur_bbx = np.array(part['bbx'])
        dis = [ (np.mean(np.square(cur_bbx - e_bbx)), e_n) for e_bbx, e_n in exist_bbx ]

        dis.sort(key=lambda x : x[0])
        if dis[0][0] > 0.02:
            print(dis)
            continue

        mesh_path = (mes_path / dis[0][1])
        point, rho = mesh_to_point(mesh_path, mes_path.stem, cur_bbx)
        part['points'] = point
        part['rho'] = rho

    with open(dst_path / (src_dat_path.stem + '.dat'), 'wb') as f:
        f.write(pickle.dumps(dat))

print("Failed:", failed)
print("Failed Count: ", len(failed))
