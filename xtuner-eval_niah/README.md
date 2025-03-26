## 📖 Single-Hop & Multi-Hop NIAH

This page provides specific evaluation methods for single-hop and multi-hop needle-in-a-haystack tasks.

---

### Preparation

**Environment**

- Please supplement the environment of VideoChat-Flash with the installation of the XTuner. For specific installation methods, please refer to [the homepage of XTuner](https://github.com/InternLM/xtuner).

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

- Please download the image data for testing from the [Huggingface link]().

**Haystack Video**

- Please place the haystack video of the model to be tested into `./xtuner/vision_niah/data/haystack_videos`. The gzyz.mkv file used in our testing can be accessed via [this link]().

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
