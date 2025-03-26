"""
# Adapted from https://huggingface.co/MILVLG/imp-v1-3b/blob/main/vision_encoder.py
"""

from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass
from functools import partial, reduce
from PIL import Image
import torch
import torch.utils.checkpoint
from torch import nn
import os
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from llava.utils import rank0_print
from .internvideo2.vit_scale_clean import PretrainVisionTransformer_clean
from .internvideo2.vit_scale_clean import interpolate_pos_embed_internvideo2

class InternVideo2ImageProcessor:
    def __init__(self, image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225), size=(224, 224), crop_size: Dict[str, int] = None, resample=PILImageResampling.BICUBIC, rescale_factor=1 / 255, data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": size[0], "width": size[1]}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors, target_size=None):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            # to adapt video data
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        if target_size is None:
            target_size = self.size
            
        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=target_size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)


class InternVideo2VisionConfig:
    model_type = "internvideo2_vision_model"

    def __init__(
        self,
        num_frames=4,
        hidden_size=1408,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_channels=3,
        image_size=224,
        patch_size=14,
        x_vis_return_idx=-2,
        sep_image_video_pos_embed=True,
        use_checkpoint=True,
        checkpoint_num=40,
        # **kwargs,
    ):
        # super().__init__(**kwargs)
        self.num_frames = num_frames
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.x_vis_return_idx = x_vis_return_idx
        self.sep_image_video_pos_embed = sep_image_video_pos_embed
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num


def build_vit(config, pt_type='origin'):

    model = PretrainVisionTransformer_clean(
        in_chans=config.num_channels, img_size=config.image_size, patch_size=config.patch_size,
        embed_dim=config.hidden_size, depth=config.num_hidden_layers, num_heads=config.num_attention_heads, mlp_ratio=48/11,
        # clip_embed_dim=config.vision_encoder.clip_embed_dim,
        attn_pool_num_heads=16, qkv_bias=False,
        drop_path_rate=0.25,
        init_values=0.00001,
        qk_normalization=True,
        use_flash_attn=True,
        use_fused_rmsnorm=False,
        use_fused_mlp=False,
        fused_mlp_heuristic=1,
        layerscale_no_force_fp32=False,
        num_frames=config.num_frames,
        tubelet_size=1,
        sep_pos_embed=False,
        sep_image_video_pos_embed=config.sep_image_video_pos_embed,
        use_checkpoint=config.use_checkpoint,
        checkpoint_num=config.checkpoint_num,
        x_vis_return_idx=config.x_vis_return_idx,
        x_vis_only=True
    )

    ckpt_path = "OpenGVLab/Video_Encoders_for_Training_VideoChat-Flash/InternVideo2-1B_f4_vision.pt"

    if not os.path.isfile(ckpt_path):
        raise NotImplementedError("Please download https://huggingface.co/OpenGVLab/Video_Encoders_for_Training_VideoChat-Flash/InternVideo2-1B_f4_vision.pt")
    
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if config.num_frames != 4:
        raise NotImplementedError
    # make deepspeed zero3 happy
    if config.image_size != 224:
        interpolate_pos_embed_internvideo2(state_dict, model, orig_t_size=4)
        
    message = model.load_state_dict(state_dict, strict=False)
    rank0_print(message)

    return model



class InternVideo2VisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False, pt_type='origin', image_size=224):
        super().__init__()

        self.is_loaded = False
        self.pt_type = pt_type

        self.config = InternVideo2VisionConfig(num_frames=vision_tower_cfg.mm_local_num_frames, x_vis_return_idx=vision_tower_cfg.mm_vision_select_layer, image_size=image_size)

        self.vision_tower_name = vision_tower

        self.image_processor = InternVideo2ImageProcessor(size=(image_size, image_size))

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            raise NotImplementedError
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.vision_tower = build_vit(self.config, pt_type=self.pt_type)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images):
        if type(images) is list:
            raise NotImplementedError
        else:
            # input: B T C H W
            # output: B T*L C
            T = images.shape[1]
            images = images.permute(0, 2, 1, 3, 4)
            image_embeds = self.vision_tower(images, use_image=(T == 1))

        return image_embeds[:, 1:, :]

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size
        # return self.model_config["vision_cfg"]["image_size"] // self.model_config["vision_cfg"]["patch_size"]

    @property
    def image_size(self):
        return self.config.image_size
