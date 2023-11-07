import math
import nltk
import torch
import wandb

from filelock import FileLock
from argparse import Namespace, ArgumentParser
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, get_scheduler
from transformers.utils import is_offline_mode

from src.dataset import ChineseNewsDataset, collate_func
from src.process import preprocess_func
from src.optimizer import get_optimizer
from src.trainer import Trainer, RLTrainer
from src.utils import set_random_seeds, read_jsonl


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Chinese News Summarization")

    parser.add_argument("--tokenizer_name", type=str,
                        default="google/mt5-small",
                        help="tokenizer name")
    parser.add_argument("--model_name_or_path", type=str,
                        default="google/mt5-small",
                        help="model name or path")
    parser.add_argument("--batch_size", type=int,
                        default=16,
                        help="batch size")
    parser.add_argument("--accum_grad_step", type=int,
                        default=4,
                        help="accumulation gradient steps")
    parser.add_argument("--epoch", type=int,
                        default=15,
                        help="number of epochs")
    parser.add_argument("--lr", type=float,
                        default=3e-5,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=0, #1e-5,
                        help="weight decay")
    parser.add_argument("--lr_scheduler", type=str,
                        default="linear",
                        help="learning rate scheduler")
    parser.add_argument("--warm_up_step", type=int,
                        default=0, #100,
                        help="number of warm up steps")
    parser.add_argument("--num_beams", type=int,
                        default=5,
                        help="number of beams search")
    parser.add_argument("--top_p", type=float,
                        default=1,
                        help="top p")
    parser.add_argument("--top_k", type=int,
                        default=0,
                        help="top k")
    parser.add_argument("--temperature", type=float,
                        default=0,
                        help="temperature")
    parser.add_argument("--device_id", type=int,
                        default=1,
                        help="deivce id")
    parser.add_argument("--use_rl", action="store_true",
                        help="Use Reinforcement Learning")
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seeds()
    args = parse_arguments()

    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        if is_offline_mode():
            raise LookupError(
                "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
            )
        with FileLock(".lock") as lock:
            nltk.download("punkt", quiet=True)

    # Prepared dataset
    train_data_list = read_jsonl("data/train.jsonl")
    valid_data_list = read_jsonl("data/public.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=True,
        trust_remote_code=False
    )
    preprocess_func = partial(preprocess_func, tokenizer=tokenizer)

    train_dataset = ChineseNewsDataset(train_data_list, preprocess_func)
    valid_dataset = ChineseNewsDataset(valid_data_list, preprocess_func)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_func, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_func, shuffle=False)

    # Prepared model
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    model_config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        config=model_config,
        trust_remote_code=False,
    ).to(device)

    # Prepared optimizer and learning rate scheduler
    optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.accum_grad_step)
    max_train_steps = args.epoch * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(args.warm_up_step / args.accum_grad_step),
        num_training_steps=max_train_steps,
    )

    # Prepared logger
    wandb.init(
        project="adl_hw2",
        name="experiment", 
        config={
            "tokenizer": args.tokenizer_name,
            "model": args.model_name_or_path,
            "epoch": args.epoch,
            "batch_size": args.batch_size,
            "accum_grad_step": args.accum_grad_step,
            "optimizer": "adamw",
            "lr_scheduler": args.lr_scheduler,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warm_up_step": args.warm_up_step,
            "num_beams": args.num_beams,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "temperature": args.temperature,
        }
    )
    wandb.watch(model, log="all")

    if args.use_rl:
        trainer = RLTrainer(
            tokenizer=tokenizer,
            model=model,
            device=device,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            accum_grad_step=args.accum_grad_step,
            lr_scheduler=lr_scheduler,
            num_beams=args.num_beams,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            logger=wandb,
        )
    else:
        trainer = Trainer(
            tokenizer=tokenizer,
            model=model,
            device=device,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            accum_grad_step=args.accum_grad_step,
            lr_scheduler=lr_scheduler,
            num_beams=args.num_beams,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            logger=wandb,
        )
    trainer.fit(epoch=args.epoch)
    wandb.finish()
