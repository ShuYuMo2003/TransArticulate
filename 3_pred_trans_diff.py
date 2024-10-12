from utils import parse_config_from_args
from lightning.pytorch import seed_everything
from model.Transformer.eval import Evaluater
from utils.mylogging import Log

from rich import print

if __name__ == '__main__':
    while True:
        try:
            config = parse_config_from_args()
            Log.info(f'Loading : {Evaluater}')
            evaluator = Evaluater(config)

            default_text = config.get('input_text')

            round = 0

            if default_text is not None:
                text = default_text
                Log.info(f'[{str(round)}] Input the prompt: {text}')
            else:
                print(f'[{str(round)}] Input the prompt: ', end='')
                text = input()
            evaluator.inference(text)
        except Exception as e:
            print(e)