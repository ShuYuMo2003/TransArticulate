import os
import re
import time
from rich import print
from glob import glob
from pathlib import Path
from tqdm import tqdm
# https://github.com/snowby666/poe-api-wrapper
from poe_api_wrapper import PoeApi

import sys
sys.path.append('..')
from sensitive_info import poe_tokens

client = PoeApi(tokens=poe_tokens)
bot_type = "gpt4_o"

prompt =  \
'''This is a type of CATE.
Please focus on its movable parts and articulation characteristics, and describe the possible motion characteristics of each part.
In the given image, there are different colored parts that can move relative to each other.
In your description, you should ignore the color, texture, and other non-structural features.

'''

length_prompt = [
    '''You can describe the motion characteristics in detail with more sentences.''',
    '''You should describe the motion characteristics with a few sentences.''',
    '''You should give me only one sentence description.''',
    '''You should give me a very short sentence containing only kind of information.''',
]

def camel_to_snake(name):
    # StorageFurniture -> storage furniture
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower()

def compress_figure(figure_path: str):
    return figure_path

def generate_description(figure_path, prompt, output_txt_path):
    print('describing fig:', figure_path, 'with', category, 'writing to', output_txt_path)

    if os.path.exists(output_txt_path):
        print('description for', figure_path, 'already exists.')
        return

    print('describing fig:', figure_path, 'with', category)
    while True:
        try:
            description = ''
            for chunk in client.send_message(
                                bot_type,
                                prompt,
                                file_path=[figure_path]):
                print(chunk["response"], end="", flush=True)
                description += chunk["response"]
            # print(description)
            if "Message too long" in description:
                print('[Error] Message too long')
                figure_path = compress_figure(figure_path)
                raise Exception('Message too long')
            break
        except Exception as e:
            print(e)
            time.sleep(2)

    print('[Write] ', output_txt_path, ": ", description)
    with open(output_txt_path, 'w') as f:
        f.write(description)

def wapper_generate_description(screenshot_path, current_output_path, category):
    current_output_path.mkdir(exist_ok=True, parents=True)
    for idx, post_prompt in enumerate(length_prompt):
        current_prompt = prompt.replace('CATE', category) + post_prompt
        output_path = current_output_path / f'{idx}.txt'
        generate_description(screenshot_path, current_prompt, output_path)

if __name__ == '__main__':
    screenshot_paths = glob('../datasets/4_screenshot_high_q/*.png')
    output_path = Path('../datasets/3_text_condition')
    Path(output_path).mkdir(exist_ok=True, parents=True)

    for screenshot_path in tqdm(screenshot_paths):
        file_name = Path(screenshot_path).stem
        key_name = file_name.split('-')[0]
        category = camel_to_snake(file_name.split('_')[0])
        current_output_path = output_path / key_name
        wapper_generate_description(screenshot_path, current_output_path, category)