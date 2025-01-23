from utils import parse_config_from_args
from lightning.pytorch import seed_everything
from model.Transformer.eval import Evaluater
from utils.mylogging import Log
from pathlib import Path

from rich import print
from tqdm import tqdm
import time
import random
import shutil

if __name__ == '__main__':
    config = parse_config_from_args()
    Log.info(f'Loading : {Evaluater}')
    evaluator = Evaluater(config)

    text_datasets = Path('data/datasets/3_text_condition')

    obj_paths = list(text_datasets.glob('*'))
    random.shuffle(obj_paths)
    tt = time.strftime("%m-%d-%I%p-%M-%S")


    OPTION = config['OPTION']
    if OPTION == -3:
        Trial = 0
        while True:
            Trial += 1
            cur_output_dir = Path("Presentation_Output") / str(Trial)
            evaluator.inference_to_output_path(input('Text Prompt:'), cur_output_dir, blender_generated_gif=False)
    if OPTION == -2:
        print("Running evaluation for result evaluation.")
        output_path = Path('data/datasets/6_ours_obj_100_same_dats')
        shutil.rmtree(output_path, ignore_errors=True)
        output_path.mkdir(exist_ok=True, parents=True)

        from evaluation_need_list import selected_for_eval_file_name
        import numpy as np
        import torch

        for trival in range(100):
            encoded_text_path = Path('/root/workspace/csm76lvhri0c73eksvjg/second_TransArticulate/data/datasets/3_encoded_text_condition/StorageFurniture_35059_4.npy')
            encoded_text = np.load(encoded_text_path, allow_pickle=True).item()['encoded_text']
            encoded_text = torch.tensor(encoded_text).type(torch.float32)
            cur_output_path = output_path / (str(trival) + '.dat')
            evaluator.inference_dat_file_only('', cur_output_path, encoded_text)

    elif OPTION == -1:
        print("Running evaluation for result evaluation.")
        output_path = Path('data/datasets/6_ours_re_obj_dats')
        shutil.rmtree(output_path, ignore_errors=True)
        output_path.mkdir(exist_ok=True, parents=True)

        from evaluation_need_list import selected_for_eval_file_name
        import numpy as np
        import torch

        for object_text_pair in selected_for_eval_file_name['fn_list']:
            object_text_pair_info = object_text_pair.split('_')
            shape_name, text_idx = '_'.join(object_text_pair_info[:2]), object_text_pair_info[2]
            encoded_text_path = Path('data/datasets/3_encoded_text_condition') / (shape_name + '_' + text_idx + ".npy")
            encoded_text = np.load(encoded_text_path, allow_pickle=True).item()['encoded_text']
            encoded_text = torch.tensor(encoded_text).type(torch.float32)

            cur_output_path = output_path / ((shape_name + '_' + text_idx) + '.dat')
            evaluator.inference_dat_file_only('', cur_output_path, encoded_text)


    elif OPTION == 0:
        import random

        random.shuffle(obj_paths)

        for obj_path in tqdm(obj_paths, 'obj count'):
            obj_name = obj_path.stem
            for text_path in tqdm(list(obj_path.glob('*')), 'text count'):
                text_name = text_path.stem
                input_text = text_path.read_text()
                output_path = Path('elog') / tt / f"{obj_name}_{text_name}"
                Log.info("Processing opt=%s ipt=%s", output_path, text_path)
                evaluator.inference_to_output_path(input_text, output_path)
    elif OPTION == 1:
        obj_name_list = ['StorageFurniture_45374_2']

        for obj_name in tqdm(obj_name_list, 'obj_list'):
            output_path = Path('elog') / f"Final_OP1_{tt}" / f"{obj_name}"
            obj_infos = obj_name.split('_')
            text_content = (text_datasets / '_'.join(obj_infos[:2]) / (str(obj_infos[2])+'.txt')).read_text()
            print("Processing", obj_name)
            for rep in range(20):
                evaluator.inference_to_output_path(text_content, output_path / str(rep), blender_generated_gif=True)
    elif OPTION == 2:

        for obj_name in ['StorageFurniture_45636_3', 'Bottle_3571_1']:
            output_path = Path('elog') / f"Final_OP1_SF_45636_3_{tt}" / f"{obj_name}"
            obj_infos = obj_name.split('_')
            text_content = (text_datasets / '_'.join(obj_infos[:2]) / (str(obj_infos[2])+'.txt')).read_text()

            for rep in range(100):
                atten_weights_list = evaluator.inference_to_output_path(text_content, output_path / str(rep), blender_generated_gif=True)

                import pickle
                with open(output_path / str(rep) / 'atten_data.pkl', 'wb') as file:
                    pickle.dump(atten_weights_list, file)
    elif OPTION == 3: # draw detailed figure

        output_path = Path('final_more_result_output') / tt
        output_path.mkdir(exist_ok=True, parents=True)
        target_obj_name = []
        for obj_path in Path('data/datasets/3_text_condition').glob('*'):
            text_paths = sorted(list(obj_path.glob('*.txt')))
            try: target_obj_name.append(obj_path.stem + "_" + text_paths[-2].stem)
            except Exception as e: pass

        import random
        random.shuffle(target_obj_name)

        for obj_name in target_obj_name:
            obj_infos = obj_name.split('_')
            text_content = (text_datasets / '_'.join(obj_infos[:2]) / (str(obj_infos[2])+'.txt')).read_text()

            for rep in range(5):
                cur_output_dir = output_path / obj_name / str(rep)
                evaluator.inference_to_output_path(text_content, cur_output_dir, blender_generated_gif=True)

    elif OPTION == 4: # Draw edit example

        target_obj_name = ['StorageFurniture_45243_0', 'StorageFurniture_45378_0', 'StorageFurniture_46029_0']
        output_path = Path('final_result_output_edit') / tt
        output_path.mkdir(exist_ok=True, parents=True)

        for obj_name in target_obj_name:
            obj_infos = obj_name.split('_')
            text_content = (text_datasets / '_'.join(obj_infos[:2]) / (str(obj_infos[2])+'.txt')).read_text()

            for rep in range(5):
                cur_output_dir = output_path / obj_name / str(rep)
                evaluator.inference_to_output_path(text_content, cur_output_dir, blender_generated_gif=True)
    else:
        raise ValueError(f"Invalid option {OPTION}")



    # python 3_pred_trans_diff.py -c configs/TF-Diff/text-eval-ours.yaml