# 👀How to train and evaluate VideoChat-Flash?🦜


## 1. Prepare Training Data


We need to address the fact that our data has been collected and used in different projects/people. For the data that has already been uploaded, we will refer you the corresponding viewing locations. Please collect relevant data fragments and integrate them in your own environments. We use similar data format with [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main/scripts/train). ***You can customize your own training data in this format***.


In [data](.data), we have provided the data used in each training stage, along with the corresponding annotation locations. We have made all the data annotations and some of the videos available on [OpenGVLab/VideoChat-Flash-Training-Data](https://huggingface.co/datasets/OpenGVLab/VideoChat-Flash-Training-Data), and I have listed all video source url in the annotation file.


## 2. Training


| Stage | Num. frames | ViT | Connector | LLM | CKPT |
|--------|:-------:|:------:|:------:|:------:|
| [stage1](scripts/train/stage1-init_connector) | 4 | :snowflake: | :fire: | :snowflake: | TBD |
| [stage2](scripts/train/stage2-visual_pretraining) | 4-8 | :fire: | :fire: | :fire: | [UMT-Qwen2_7B](https://huggingface.co/OpenGVLab/stage2-UMT-Qwen2-7B-tome16_mlp), [UMT-Qwen2_5_1M_7B](https://huggingface.co/OpenGVLab/stage2-UMT-Qwen2_5_7B_1m-tome16_mlp), [UMT-HD-Qwen2_5_2B](https://huggingface.co/OpenGVLab/stage2-UMT-Qwen2_5_1.5B-tome16_mlp), [InternVideo2-Qwen2_5_7B](https://huggingface.co/OpenGVLab/stage2-InternVideo2-1B-Qwen2_5-7B-tome16_mlp) |
| [stage3](scripts/train/stage3-video_sft) | 64-512 | :fire: | :fire: | :fire: | [UMT-Qwen2_7B](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2-7B_res448),[UMT-HD-Qwen2_5-2B](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448),[UMT-Qwen2_5_1M_7B](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-7B-1M_res224), [InternVideo2-Qwen2_5_7B](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B) |
| [stage4](scripts/train/stage4_highres_postft) | 64-512 | :fire: | :fire: | :snowflake: | [UMT-HD-Qwen2-7B](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2-7B_res448)|

### Install

```bash
git clone https://github.com/OpenGVLab/VideoChat-Flash
cd llava-train_videochat
pip install -e .
```

### Stage-1: Video-Language Alignment

Please download pretrained video encoders in [Huggingfaces](https://huggingface.co/OpenGVLab/Video_Encoders_for_Training_VideoChat-Flash) first. Then modify ckpt_path in `build_vit` of `llava/model/multimodal_encoder/umt_encoder.py` or `llava/model/multimodal_encoder/internvideo2_encoder.py`.
```bash
bash scripts/train/stage1-init_connector/stage1_umt_tome16_res224_qwen7b.sh
```
### Stage-2: Short Video Pre-training
```bash
bash scripts/train/stage2-visual_pretraining/stage2_umt_tome16_res224_qwen_7b.sh
```
### Stage-3: Joint Short & Long Video Instruction Tuning
```bash
bash scripts/train/stage3-video_sft/stage3_umt_tome16_res224_qwen_7b.sh
```

### Stage-4: Efficient High-Resolution Post-finetuning
Please modify `vision_tower="umt-hd-large"` in `Your_stage3_checkpoint_path/config.json` first!
```bash
bash scripts/train/stage4_highres_postft/stage4_umt_tome16_res448_qwen_7b.sh
```

## Evaluation

Overwrite your checkpoints directory with the configurations and Python files from OpenGVLab/VideoChat-Flash, and then you can use the lmms-eval_videochat we provided for evaluation.
