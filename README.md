# Chinese News Summarization
This repository is implementation of Homework 2 for CSIE5431 Applied Deep Learning course in 2023 Fall semester at National Taiwan University.


## Setting the Environment
**Step 1:** Install the main packages
To set the environment, you can run this command:
```
pip install -r config/requirements.txt
```

**Step 2:** Install the evaluation metric
To install the evaluation metric, you can run this command:
```
git clone https://github.com/moooooser999/ADL23-HW2.git
cd ADL23-HW2
pip install -e tw_rouge
```


## Download dataset and model checkpoint
To download the datasets and model checkpoint, you can run the commad:
```
bash ./download.sh
```

## Reproducing best result
To reproduce our best result, you can run the commad:
```
bash ./run.sh <data file> <output file>
```


## Training
To train the summarization model, you can run the commad:
```
python train.py
```


## Testing
To test the summarization model, you can run the commad:
```
python test.py
```


## Evaluating Submission
To evaluate the submission file, you can run the command
```
python ADL23-HW2/eval.py -r public.jsonl -s submission.jsonl
```


## Experiment Results
<table>
  <tr>
    <td>Method</td>
    <td>Rouge-1</td>
    <td>Rouge-2</td>
    <td>Rouge-L</td>
  </tr>
  <tr>
    <td>Our</td>
    <td>26.8510</td>
    <td>10.7346</td>
    <td>23.9712</td>
  </tr>
  <tr>
    <td>Baseline</td>
    <td>22.0</td>
    <td>8.5</td>
    <td>20.5</td>
  </tr>
<table>


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
