## 📖 Single-Hop & Multi-Hop NIAH

This page provides specific evaluation methods for single-hop and multi-hop needle-in-a-haystack tasks.

---

### Preparation

**Environment**

- Please use a environment with **transformers >= 4.45.1 (to use YARN)**. see [niah_requirements.txt](niah_requirements.txt).

- And to avoid confict, **please don't install XTuner in your environment**, I have copy all useful files in [xtuner](xtuner).

**Model Parameters**

- Please place the pretrained parameters of the model to be tested into `./xtuner/vision_niah/model_weights/`.

- Please pay extra attention to modifying the `"rope_scaling": null` in `VideoChat-Flash-Qwen2-7B_res448/config.json` to:

```
"rope_scaling": {
    "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768
  }
```

**Eval Datasets**

- Please download the image and videodata for testing from the [Huggingface link](https://huggingface.co/datasets/OpenGVLab/NIAH-Video).
- We place the annotaions of NIAH in `vision_niah/data` and `vision_niah/data_multi`.

**Haystack Video**

- Please place the haystack video of the model to be tested into `./xtuner/vision_niah/data/haystack_videos`. The video_haystack.mkv file used in our testing can be accessed via [this link](https://huggingface.co/datasets/OpenGVLab/NIAH-Video).

---

### Eval

- When running the needle-in-a-haystack test, please set the `xtuner_eval_niah` folder as the root directory of the workspace.

**single-hop niah**

- Please run the following command.
```
bash vision_niah/flash_eval_xtuner_single.sh
```

**multi-hop niah**

- Please run the following command.
```
bash vision_niah/flash_eval_xtuner_multi.sh
```
