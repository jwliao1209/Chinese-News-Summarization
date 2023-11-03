import json
import math
import torch

from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, get_scheduler

from src.constants import MAX_SOURCE_LEN, MAX_TARGET_LEN, SUMMARY_COL
from src.optimizer import get_optimizer
from src.trainer import Trainer
from src.utils import read_jsonl


tokenizer_name = "google/mt5-small"
model_name_or_path = "google/mt5-small"
batch_size = 8
lr = 1e-3
weight_decay = 0
lr_scheduler = "cosine"
warm_up_step = 0
accum_grad_step = 4
epoch = 15


TEXT_COL = "maintext"
SUMMARY_COL = "title"

train_data_list = read_jsonl("data/data/train.jsonl")
valid_data_list = read_jsonl("data/data/public.jsonl")

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name,
    use_fast=True,
    trust_remote_code=False
)

model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=False)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name_or_path,
    config=model_config,
    trust_remote_code=False,
)


def preprocess_func(data, tokenizer, train=True):
    tokenized_data = tokenizer(data[TEXT_COL], max_length=MAX_SOURCE_LEN, padding="max_length", truncation=True)
    label = tokenizer(text_target=data[SUMMARY_COL], max_length=MAX_TARGET_LEN, padding="max_length", truncation=True)["input_ids"]
    label = [(l if l != tokenizer.pad_token_id else -100) for l in label]
    if train:
        tokenized_data[SUMMARY_COL] = data[SUMMARY_COL]
        tokenized_data["labels"] = label
    return tokenized_data


def postprocess_func(batch_data):
    return ["\n".join(nltk.sent_tokenize(data.strip())) for data in batch_data]


def collate_func(data):
    # list of dict -> dict of list
    data = {k: [dic[k] for dic in data] for k in data[0]}
    data = {k: v if k in [SUMMARY_COL] else torch.tensor(v)
            for k, v in data.items()}
    return data

class ChineseNewsDataset(Dataset):
    def __init__(self, data_list, transform=False):
        self.data_list = [transform(data) for data in tqdm(data_list)] if transform is not None else data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]
    
    
preprocess_func = partial(preprocess_func, tokenizer=tokenizer)
train_dataset = ChineseNewsDataset(train_data_list, preprocess_func)
valid_dataset = ChineseNewsDataset(valid_data_list, preprocess_func)


train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_func, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_func, shuffle=False)


device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")


optimizer = get_optimizer(
    model, lr=lr, weight_decay=weight_decay
)


num_update_steps_per_epoch = math.ceil(len(train_loader) / accum_grad_step)
max_train_steps = epoch * num_update_steps_per_epoch
lr_scheduler = get_scheduler(
    name=lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=math.ceil(warm_up_step / accum_grad_step),
    num_training_steps=max_train_steps,
)


trainer = Trainer(
    tokenizer=tokenizer,
    model=model,
    device=device,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=optimizer,
    accum_grad_step=accum_grad_step,
    lr_scheduler=lr_scheduler,
)


trainer.fit(epoch=epoch)

