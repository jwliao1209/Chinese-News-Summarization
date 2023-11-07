import numpy as np
import torch
import torch.nn as nn
from src.metric import RougeScore


class PolicyGradientLoss(nn.Module):
    def __init__(
        self,
        weight=torch.Tensor([0.3, 0.4, 0.3]),
        baseline=torch.Tensor([0.220, 0.085, 0.205]),
        device=None,
        ):
        super(PolicyGradientLoss, self).__init__()
        self.weight = weight
        self.baseline = baseline
        self.device = device
        self.rouge_score_func = RougeScore()
        self.ce_loss_func = torch.nn.CrossEntropyLoss(reduction="none")

    def get_scores(self, generations, references):
        rouge_scores = self.rouge_score_func.evaluate(
                generations, references, avg=False,
        )
        score = torch.tensor([
            [score["rouge-1"] * 100 for score in rouge_scores],
            [score["rouge-2"] * 100 for score in rouge_scores],
            [score["rouge-l"] * 100 for score in rouge_scores],
        ]).reshape(-1, 3)
        return score
    
    def get_reward(self, generations, references):
        scores = self.get_scores(generations, references)
        return (self.weight.unsqueeze(0) * scores).sum(dim=1)

    def forward(self, logits, labels, generations, references):
        reward = self.get_reward(generations, references).unsqueeze(0).to(self.device)
        ce_loss = self.ce_loss_func(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1)
        ).view(logits.shape[0], -1).mean(dim=1)
        return (reward * ce_loss).mean()
