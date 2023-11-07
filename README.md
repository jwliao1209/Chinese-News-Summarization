# Chinese News Summarization
This repository is implementation of Homework 2 for CSIE5431 Applied Deep Learning course in 2023 Fall semester at National Taiwan University.


## Setting the Environment
To set the environment, you can run this command:
```
pip install -r requirements.txt
```


## Download dataset and model checkpoint
To download the datasets and model checkpoint, you can run the commad:
```
bash ./download.sh
```

## Reproducing best result
To reproduce our best result, you can run the commad:
```
bash ./run.sh data/public.jsonl pred/output.jsonl
```


## Training
```
bash ./train.sh
```


## Ｔesting
```
python test.py
```

## Inference
```
python infer.py
```


## Evaluation Submission
```
git clone https://github.com/moooooser999/ADL23-HW2.git
cd ADL23-HW2
pip install -e tw_rouge

python eval.py -r public.jsonl -s submission.jsonl
```



## Experiment Results



## Operating System and Device
We implemented the code on an environment running Ubuntu 22.04.1, utilizing a 12th Generation Intel(R) Core(TM) i7-12700 CPU, along with a single NVIDIA GeForce RTX 4090 GPU equipped with 24 GB of dedicated memory.


## Acknowledgement
We thank the Hugging Face repository: https://github.com/huggingface/transformers


## Citation
```bibtex
@misc{
    title  = {2023_adl_hw2_chinese_news_summarization},
    author = {Jia-Wei Liao},
    url    = {https://github.com/jwliao1209/Chinese-News-Summarization},
    year   = {2023}
}
```
