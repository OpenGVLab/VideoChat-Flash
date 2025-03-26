import os
from .clip_encoder import CLIPVisionTower
from .siglip_encoder import SigLipVisionTower
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .umt_encoder import UMTVisionTower
from .internvideo2_encoder import InternVideo2VisionTower
# from .eva_clip.eva_clip_encoder import EvaClipVisionTower
# from .dev_eva_clip.eva_vit import EvaViTWrapper


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    # is_absolute_path_exists = os.path.exists(vision_tower) # NOTE sb code!
    use_s2 = getattr(vision_tower_cfg, "s2", False)
    if 'clip-vit' in vision_tower or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "siglip" in vision_tower:
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    elif "internvideo2" in vision_tower:
        return InternVideo2VisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, image_size=224, **kwargs)
    elif "umt-hd" in vision_tower:
        return UMTVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, image_size=448, **kwargs)
    elif "umt" in vision_tower:
        return UMTVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f"Unknown vision tower: {vision_tower}")
