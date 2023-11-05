import os
import gdown
import zipfile

from rouge import Rouge
from ckiptagger import WS


def download_data_gdown(path):
    file_id = "1efHsY16pxK0lBD2gYCgCTnv1Swstq771"
    url = f"https://drive.google.com/uc?id={file_id}"
    data_zip = os.path.join(path, "data.zip")
    gdown.download(url, data_zip, quiet=False)
    with zipfile.ZipFile(data_zip, "r") as zip_ref:
        zip_ref.extractall(path)
    return


def get_ws():
    cache_dir = os.environ.get("XDG_CACHE_HOME", os.path.join(os.getenv("HOME"), ".cache"))
    download_dir = os.path.join(cache_dir, "ckiptagger")
    data_dir = os.path.join(cache_dir, "ckiptagger/data")
    os.makedirs(download_dir, exist_ok=True)
    if not os.path.exists(os.path.join(data_dir, "model_ws")):
        download_data_gdown(download_dir)
    return WS(data_dir, disable_cuda=False)


class RougeScore:
    def __init__(self):
        self.rouge = Rouge()
        self.ws = get_ws()

    def tokenize_and_join(self, sentences):
        return [" ".join(toks) for toks in self.ws(sentences)]

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
