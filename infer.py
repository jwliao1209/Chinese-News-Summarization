import torch

from tqdm import tqdm
from argparse import Namespace, ArgumentParser
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM

from src.constants import MAX_TARGET_LEN, SUMMARY_COL
from src.dataset import ChineseNewsDataset, collate_func
from src.process import preprocess_func, postprocess_func
from src.utils import set_random_seeds, read_jsonl, dict_to_device, write_jsonl


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Chinese News Summarization")
    parser.add_argument("--data_path", type=str,
                        default="data/public.jsonl",
                        help="data path")
    parser.add_argument("--tokenizer_name", type=str,
                        default="google/mt5-small",
                        help="tokenizer name")
    parser.add_argument("--model_name_or_path", type=str,
                        default="checkpoint_1/epoch=15_rouge-1=26.850953064812938",
                        help="model name or path")
    parser.add_argument("--batch_size", type=int,
                        default=100,
                        help="batch size")
    parser.add_argument("--num_beams", type=int,
                        default=5,
                        help="number of beams search")
    parser.add_argument("--do_sample", action="store_true",
                        help="do sampling stratgies")
    parser.add_argument("--top_p", type=float,
                        default=0,
                        help="top p")
    parser.add_argument("--top_k", type=int,
                        default=0,
                        help="top k")
    parser.add_argument("--temperature", type=float,
                        default=0,
                        help="temperature")
    parser.add_argument("--device_id", type=int,
                        default=0,
                        help="deivce id")
    parser.add_argument("--output_path", type=str,
                        default="pred/output.jsonl",
                        help="output file")
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seeds()
    args = parse_arguments()

    # Prepared dataset
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=True,
        trust_remote_code=False
    )
    test_data_list = read_jsonl(args.data_path)
    preprocess_func = partial(preprocess_func, tokenizer=tokenizer, train=False)
    test_dataset = ChineseNewsDataset(test_data_list, preprocess_func)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_func, shuffle=False)

    # Prepared model
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    model_config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        config=model_config,
        trust_remote_code=False,
    ).to(device)

    sampling_params = {
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        if args.top_p > 0:
            sampling_params["top_p"] = args.top_p
        if args.top_k > 0:
            sampling_params["top_k"] = args.top_k
        if args.temperature > 0:
            sampling_params["temperature"] = args.temperature
    print(sampling_params)

    model.eval()
    prediction_list = []
    test_bar = tqdm(test_loader, desc=f"Testing")
    for _, batch_data in enumerate(test_bar, start=1):
        with torch.no_grad():
            batch_data = dict_to_device(batch_data, device)
            generated_tokens = model.generate(
                input_ids=batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=MAX_TARGET_LEN,
                num_beams=args.num_beams,
                **sampling_params,
            )
            generations = postprocess_func(
                tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            )
            prediction_list.extend(
                [
                    {SUMMARY_COL: pred, "id": ID}
                    for ID, pred in zip(batch_data["id"], generations)
                ]
            )
    write_jsonl(prediction_list, args.output_path)
