import json
import nltk
import random
import torch
import numpy as np
from filelock import FileLock
from transformers.utils import is_offline_mode


def read_jsonl(file_path: str) -> list:
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def write_jsonl(data_list: list, path: str) -> None:
    with open(path, "w") as fp:
        for data in data_list:
            fp.write(json.dumps(data, ensure_ascii=False))
            fp.write('\n')
    return


def set_random_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return


def prepare_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        if is_offline_mode():
            raise LookupError(
                "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
            )
        with FileLock(".lock") as lock:
            nltk.download("punkt", quiet=True)
    return


def dict_to_device(data: dict, device: torch.device) -> dict:
    return {k: v.to(device) if not isinstance(v, list) else v for k, v in data.items()}
