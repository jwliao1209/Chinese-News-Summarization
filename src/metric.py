import os
from rouge import Rouge
from ckiptagger import WS, data_utils


cache_dir = os.environ.get("XDG_CACHE_HOME", os.path.join(os.getenv("HOME"), ".cache"))
download_dir = os.path.join(cache_dir, "ckiptagger")
data_dir = os.path.join(cache_dir, "ckiptagger/data")
os.makedirs(download_dir, exist_ok=True)

if not os.path.exists(os.path.join(data_dir, "model_ws")):
    data_utils.download_data_gdown(download_dir)

ws = WS(data_dir, disable_cuda=False)


class RougeScore:
    def __init__(self):
        self.rouge = Rouge()

    def tokenize_and_join(self, sentences):
        return [" ".join(toks) for toks in ws(sentences)]

    def evaluate(self, preds, refs, avg=True, ignore_empty=False):
        """wrapper around: from rouge import Rouge
        Args:
            preds: string or list of strings
            refs: string or list of strings
            avg: bool, return the average metrics if set to True
            ignore_empty: bool, ignore empty pairs if set to True
        """
        preds = self.tokenize_and_join([preds] if not isinstance(preds, list) else preds)
        refs = self.tokenize_and_join([refs] if not isinstance(refs, list) else refs)
        scores = self.rouge.get_scores(preds, refs, avg=avg, ignore_empty=ignore_empty)
        return {k: v["f"] * 100 for k, v in scores.items()}
