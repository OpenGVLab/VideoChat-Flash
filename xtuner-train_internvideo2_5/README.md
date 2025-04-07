#  How to finetuning InternVideo2.5?

Note: We only support the training with **video data**.
## Install

```
cd xtuner-train_internvideo2_5
pip install -e .
```

## Prepare your data

1. Prepare your data annotations like [this](data/annotaions/ft_data_example.jsonl), if you need to use data packing, all data item in annotation file shuld have `duration`.
2. List your training data in `data/diy_ft_data.json`.

## Start to training

If you need to use data packing to speed up:
```bash
bash ft_internvideo_2_5_datapacking.sh
```
otherwise:
```bash
bash ft_internvideo_2_5.sh
```

## Evaluation

Copy the python file in https://huggingface.co/OpenGVLab/InternVideo2_5_Chat_8B and use [lmms_eval](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/models/internvideo2_5.py) to evaluate.
