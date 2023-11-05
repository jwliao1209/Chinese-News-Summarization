import torch
from src.metric import RougeScore


class RLLoss:
    def __init__(
        self,
        weight=torch.Tensor([0.3, 0.4, 0.3]),
        baseline=torch.Tensor([0.220, 0.085, 0.205]),
        device=None,
        ):
        self.weight = weight
        self.baseline = baseline
        self.device = device
        self.rouge_score_func = RougeScore()
        self.ce_loss_func = torch.nn.CrossEntropyLoss(reduction="none")
    
    def get_scores(self, generations, references):
        rouge_scores = self.rouge_score_func.evaluate(
                generations, references, avg=False,
        )
        return torch.tensor(
            [rouge_scores[k] / 100 for k in ["rouge-1", "rouge-2", "rouge-l"]]
        )
    
    def get_reward(self, generations, references):
        scores = self.get_scores(generations, references)
        return torch.sum(self.weight * scores / self.baseline, axis=1)

    def forward(self, logits, labels, generations, references, weight):
        reward = self.get_reward(generations, references).to(self.device)
        ce_loss = self.ce_loss_func(logits, labels).mean(dim=1)
        return (reward * ce_loss).mean()
