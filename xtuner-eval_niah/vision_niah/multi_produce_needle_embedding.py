import sys
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path
from PIL import Image
from datasets import load_dataset
import math
import cv2
import io
from petrel_client.client import Client
client = Client('~/petreloss.conf')


data_root_path = "pnorm2:s3://coco-caption/"


def main(args):
    if "videochat-flash" in args.model.lower():
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
        from llava.mm_utils import  process_images
    elif "longva" in args.model.lower():
        from longva.model.builder import load_pretrained_model
        from longva.mm_utils import tokenizer_image_token, get_model_name_from_path
        from longva.mm_utils import  process_images
    else:
        raise "This version model is not currently supported. Please manually adjust the code to adapt."
    
    
    model_name = "llava_qwen"
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model, None, model_name, load_8bit=False,device_map="cuda:0")
    model.config.image_aspect_ratio = "pad"
    model.config.mm_patch_merge_type="flat"
    
    
    with open(args.needle_dataset, 'r', encoding='utf-8') as file: 
        data = json.load(file)
    
    
    for index, instance in enumerate(tqdm(data, desc="Processing")):
        images_list = []
        for image_path in instance["images"]:
            frame_path = os.path.join(data_root_path, image_path)
            if "s3://" in data_root_path:
                img_bytes = client.get(frame_path)
            else:
                with open(frame_path, 'rb') as f:
                    img_bytes = f.read()
            # img_np = np.frombuffer(img_bytes, np.uint8)
            # img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            # cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            # images_list.append(img)
            # 使用 PIL 来处理图像
            img = Image.open(io.BytesIO(img_bytes))
            img = img.convert("RGB")
            images_list.append(img)
            
        # print(images_list.shape)
        
        images_list = process_images(images_list, image_processor, model.config).half()
        
        # print("image before encode:", images_list.shape)
        if "videochat-flash" in args.model.lower():
            batch_size = 4
            image_features_list=[]
            for idx in range(images_list.shape[0]):
                processed_images = images_list[idx:(idx+1)].repeat(batch_size, 1, 1, 1)
                # print("processed_images:", processed_images.shape)
                image_features = model.encode_video_image([processed_images], video_idx_in_batch = [0])[0]
                # print("image_features:", image_features.shape)
                image_features_list.append(image_features)
            image_features=torch.cat(image_features_list,dim=0)
        else:
            image_features = model.encode_images(images_list)
        # print("needle embedding shape: ", image_features.shape)
        if args.pooling_size != 0:
            B, _, F = image_features.shape
            image_features_spatial = image_features.view(B, int(math.sqrt(_)), int(math.sqrt(_)), F).permute(0, 3, 1, 2) # B, F, 24, 24
            image_features_spatial_pool = torch.nn.functional.avg_pool2d(image_features_spatial, args.pooling_size, args.pooling_size) # B, F, 12, 12
            image_features = image_features_spatial_pool.flatten(2).transpose(1, 2).contiguous() # B, 144, F
        image_features = image_features.squeeze(0)
        # print("needle shape after pooling: ", image_features.shape)
        if "videochat-flash" not in args.model.lower():
            image_features = image_features.repeat(1, 4, 1)
        print("final save needle shape: ", image_features.shape)
        torch.save(image_features, f"{args.output_dir}/{index}.pt")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="output/LLaVA-NeXT-Video-7B-Vicuna")
    parser.add_argument("--needle_dataset", type=str, default="vision_niah/data_multi/source_data/niah-coco-multihop_mc.json")
    parser.add_argument("--output_dir", type=str, default="video_needle_haystack/data/needle_vicuna_embeddings")
    parser.add_argument("--pooling_size", type=int, default=0)
    args = parser.parse_args()
    main(args)
