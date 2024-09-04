from utils import parse_config_from_args
from lightning.pytorch import seed_everything
from model.TransformerDiffusion.eval import Evaluater

if __name__ == '__main__':
    config = parse_config_from_args()

    evaluator = Evaluater(config)

    round = 0
    while True:
        text = input(f'[{str(round)}] Input the prompt: ')
        evaluator.inference(text, round)
        round += 1