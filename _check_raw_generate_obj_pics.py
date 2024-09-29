import json
import trimesh
from pathlib import Path

from rich import print

from utils import generate_gif_toy


def main():
    obj_info = json.loads(Path('data/datasets/1_preprocessed_info/Bottle_3380.json').read_text())

    tokens = obj_info['part']

    for token in tokens:
        # print(token)
        mesh_path = Path('data/datasets/1_preprocessed_mesh') / token['mesh']
        mesh = trimesh.load(mesh_path.as_posix())
        token['mesh'] = mesh

    generate_gif_toy(tokens, Path('log/qaq.gif'), n_frame=20)


if __name__ == '__main__':
    main()