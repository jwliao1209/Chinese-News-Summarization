import json
import nltk
import random
import torch
import numpy as np


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def save_json(obj: dict, path: str) -> None:
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=4)
    return


def set_random_seeds(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return


def dict_to_device(data: dict, device: torch.device) -> dict:
    return {k: v.to(device) if not isinstance(v, list) else v for k, v in data.items()}
