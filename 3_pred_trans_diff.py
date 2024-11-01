from utils import parse_config_from_args
from lightning.pytorch import seed_everything
from model.Transformer.eval import Evaluater
from utils.mylogging import Log
from pathlib import Path

from rich import print
from tqdm import tqdm
import time
import random

if __name__ == '__main__':
    config = parse_config_from_args()
    Log.info(f'Loading : {Evaluater}')
    evaluator = Evaluater(config)

    text_datasets = Path('data/datasets/3_text_condition')

    obj_paths = list(text_datasets.glob('*'))
    random.shuffle(obj_paths)
    tt = time.strftime("%m-%d-%I%p-%M-%S")


    OPTION = 2

    if OPTION == 0:
        for obj_path in tqdm(obj_paths, 'obj count'):
            obj_name = obj_path.stem
            for text_path in tqdm(list(obj_path.glob('*')), 'text count'):
                text_name = text_path.stem
                input_text = text_path.read_text()
                output_path = Path('elog') / tt / f"{obj_name}_{text_name}"
                Log.info("Processing opt=%s ipt=%s", output_path, text_path)
                evaluator.inference_to_output_path(input_text, output_path)
    elif OPTION == 1:
        obj_name_list = ['StorageFurniture_45243_1',
                         'StorageFurniture_45374_2', 'StorageFurniture_45636_3',
                         'StorageFurniture_45710_0', 'StorageFurniture_46466_2',
                         'StorageFurniture_46856_0', 'StorageFurniture_46856_1']

        for obj_name in tqdm(obj_name_list, 'obj_list'):
            output_path = Path('elog') / f"Final_OP1_{tt}" / f"{obj_name}"
            obj_infos = obj_name.split('_')
            text_content = (text_datasets / '_'.join(obj_infos[:2]) / (str(obj_infos[2])+'.txt')).read_text()
            print("Processing", obj_name)
            for rep in range(20):
                evaluator.inference_to_output_path(text_content, output_path / str(rep), blender_generated_gif=True)
    elif OPTION == 2:

        for obj_name in ['StorageFurniture_45632_1', 'StorageFurniture_45374_2', 'StorageFurniture_45622_1',
                         'StorageFurniture_46109_2', 'USB_85_1', 'USB_2062_1', 'Bottle_3571_1']:
            output_path = Path('elog') / f"Final_OP1_SF_45632_{tt}" / f"{obj_name}"
            obj_infos = obj_name.split('_')
            text_content = (text_datasets / '_'.join(obj_infos[:2]) / (str(obj_infos[2])+'.txt')).read_text()

            for rep in range(100):
                atten_weights_list = evaluator.inference_to_output_path(text_content, output_path / str(rep), blender_generated_gif=True)

                import pickle
                with open(output_path / str(rep) / 'atten_data.pkl', 'wb') as file:
                    pickle.dump(atten_weights_list, file)
