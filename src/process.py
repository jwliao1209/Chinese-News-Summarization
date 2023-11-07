import nltk
from src.constants import TEXT_COL, SUMMARY_COL, MAX_SOURCE_LEN, MAX_TARGET_LEN


def preprocess_func(data, tokenizer, train: bool = True):
    tokenized_data = tokenizer(data[TEXT_COL], max_length=MAX_SOURCE_LEN, padding="max_length", truncation=True)
    label = tokenizer(text_target=data[SUMMARY_COL], max_length=MAX_TARGET_LEN, padding="max_length", truncation=True)["input_ids"]
    label = [(l if l != tokenizer.pad_token_id else -100) for l in label]
    if train:
        tokenized_data[SUMMARY_COL] = data[SUMMARY_COL]
        tokenized_data["labels"] = label
    else:
        tokenized_data["id"] = data["id"]
    return tokenized_data
    

def postprocess_func(batch_data):
    return ["\n".join(nltk.sent_tokenize(data.strip())) for data in batch_data]
