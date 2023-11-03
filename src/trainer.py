import os
import torch
import evaluate
from tqdm import tqdm
from src.constants import CHECKPOINT_DIR, SUMMARY_COL, MAX_TARGET_LEN
# from src.metric import RougeScore
from src.process import postprocess_func
from src.tracker import MetricTracker
from src.utils import dict_to_device, postprocess_func


class Trainer:
    def __init__(
        self,
        tokenizer,
        model,
        device,
        train_loader,
        valid_loader,
        optimizer,
        accum_grad_step,
        lr_scheduler,
        logger=None,
        num_beams=5,
        top_p=1,
        top_k=0,
        temperature=1,
        *arg,
        **kwarg,
        ):
        
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_num = len(train_loader)
        self.valid_num = len(valid_loader)
        self.optimizer = optimizer
        self.accum_grad_step = accum_grad_step
        self.lr_scheduler = lr_scheduler
        # self.eval_func = RougeScore()
        self.tracker = MetricTracker()
        self.logger = logger

        # generation arguments
        self.num_beams = num_beams
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature

    def train_step(self, batch_data, index):
        outputs = self.model(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            labels=batch_data["labels"],
        )
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        n = preds.shape[0]
        self.tracker.update("train/loss", loss / n, n)
        return loss

    def valid_step(self, batch_data, index):
        generated_tokens = self.model.generate(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            max_length=MAX_TARGET_LEN,
            num_beams=self.num_beams,
            do_sample=False, 
            top_p=self.top_p, 
            top_k=self.top_k,
            temperature=self.temperature,
        )
        return generated_tokens

    def log(self, record):
        # self.progress_bar.set_postfix(record)
        if self.logger is not None:
            self.logger.log(record)
        return

    def train_one_epoch(self):
        self.model.train()
        self.progress_bar = tqdm(self.train_loader, desc=f"Training {self.cur_ep}")
        self.tracker.reset(keys=["train/loss"])

        for step, batch_data in enumerate(self.progress_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)
            loss = self.train_step(batch_data, step)
            self.progress_bar.set_postfix({**self.tracker.result(), "lr": self.lr_scheduler.get_last_lr()[0]})
            self.log({**self.tracker.result(), "lr": self.lr_scheduler.get_last_lr()[0]})

            (loss / self.accum_grad_step).backward()
            if step % self.accum_grad_step == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

        self.progress_bar.close()
        return

    @torch.no_grad()
    def valid_one_epoch(self):
        self.model.eval()
        self.progress_bar = tqdm(self.valid_loader, desc=f"Validation {self.cur_ep}")
        self.tracker.reset(keys=["valid/loss", "valid/rouge1", "valid/rouge2", "valid/rougeL"])

        # prediction_list = []
        # ground_truth_list = []

        metric = evaluate.load("rouge")

        for step, batch_data in enumerate(self.progress_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)
            generated_tokens = self.valid_step(batch_data, step)
            generated_tokens = postprocess_func(self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
            ground_truth = batch_data[SUMMARY_COL]
            metric.add_batch(predictions=generated_tokens, references=ground_truth)

        result = metric.compute(use_stemmer=True)
        score = {k: round(v * 100, 2) for k, v in result.items()}
        print(score)

        # score = self.eval_func.evaluate(prediction_list, ground_truth_list)
        for rouge in ["rouge1", "rouge2", "rougeL"]:
            self.tracker.update(f"valid/{rouge}", score[rouge])
            self.tracker.update(f"valid/{rouge}", score[rouge])

        self.log({"epoch": self.cur_ep, **self.tracker.result()})
        self.progress_bar.close()
        self.model.save_pretrained(
            os.path.join(
                CHECKPOINT_DIR,
                f"epoch={self.cur_ep}_rouge-1={self.tracker.result().get('valid/rouge1', 0)}"
            )
        )
        return

    def fit(self, epoch):
        self.model.to(self.device)
        for self.cur_ep in range(1, epoch+1):
            self.train_one_epoch()
            self.valid_one_epoch()
        return
