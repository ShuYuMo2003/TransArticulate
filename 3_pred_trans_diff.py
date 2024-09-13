from utils import parse_config_from_args
from lightning.pytorch import seed_everything
from model.Transformer.eval import Evaluater
from utils.logging import Log

from rich import print

if __name__ == '__main__':
    config = parse_config_from_args()
    Log.info(f'Loading : {Evaluater}')
    evaluator = Evaluater(config)

    default_text = config.get('input_text')

    round = 0
    while True:
        if default_text is not None:
            text = default_text
            Log.info(f'[{str(round)}] Input the prompt: {text}')
        else:
            print(f'[{str(round)}] Input the prompt: ', end='')
            text = input()
        evaluator.inference(text, round)
        round += 1
        # break