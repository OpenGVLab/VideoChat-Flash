
import sys
import os
from decord import VideoReader, cpu
import argparse
import numpy as np
from tqdm import tqdm
import torch
import math
from PIL import Image

def load_video_batches(video_path, batch_size):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    for start_idx in range(0, len(frame_idx), batch_size):
        end_idx = min(start_idx + batch_size, total_frame_num)
        frame_indices = frame_idx[start_idx:end_idx]
        batch_frames = vr.get_batch(frame_indices).asnumpy()
        yield batch_frames

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
    
    
    video_path = args.video_path
    model_path = args.model
    model_name = "llava_qwen"
    
    if args.sampled_frames_num > 7500:
        sample_frame_num = args.sampled_frames_num // 2 
        double_flag = True
    else:
        sample_frame_num = args.sampled_frames_num
        double_flag = False
        
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False,device_map="cuda:0")
    del model.model.layers
    model.config.image_aspect_ratio = "pad"
    model.config.mm_patch_merge_type="flat"
    # Process video in batches
    batch_size = 8
    total_batches = (sample_frame_num + batch_size - 1) // batch_size
    image_feature_list = []
    if args.add_newline_token:
        newline_token_embeddong = model.model.image_newline
    with torch.inference_mode():
        for i, video_batch in tqdm(enumerate(load_video_batches(video_path, batch_size)), total=total_batches, desc="Processing Video Batches"):
            images = [Image.fromarray(frame).convert("RGB") for frame in video_batch]
            processed_images = process_images(images, image_processor,model.config).half()
            # print("processed_images.shape",processed_images.shape)
            
            if "videochat-flash" in model_path.lower():
                image_features = model.encode_video_image([processed_images], video_idx_in_batch = [0] )[0]
                image_features = image_features.reshape(batch_size, -1, image_features.shape[-1])
            else:
                image_features = model.encode_images(processed_images)
            
            # print(image_features.shape)
            
            if args.pooling_size != 0:
                B, _, F = image_features.shape
                
                image_features_spatial = image_features.view(B, int(math.sqrt(_)), int(math.sqrt(_)), F).permute(0, 3, 1, 2) # B, F, 24, 24
                image_features_spatial_pool = torch.nn.functional.avg_pool2d(image_features_spatial, args.pooling_size, args.pooling_size) # B, F, 12, 12
                image_features = image_features_spatial_pool.flatten(2).transpose(1, 2).contiguous() # B, 144, F
            if args.add_newline_token:
                image_features = torch.cat([image_features, newline_token_embeddong.unsqueeze(0).expand(image_features.shape[0], 1, -1)], dim=1)
            image_feature_list.append(image_features.to(torch.bfloat16).to("cpu"))
            if i > total_batches:
                break
    image_feature_list = torch.cat(image_feature_list, dim=0)
    if double_flag:
        image_feature_list = torch.cat((image_feature_list, image_feature_list), dim=0)
    print("final save haystack shape: ", image_feature_list.shape)
    torch.save(image_feature_list, f"{args.output_dir}/video_embeddings.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="output/LLaVA-NeXT-Video-7B-Vicuna")
    parser.add_argument("--video_path", type=str, default="vision_niah/data/haystack_videos")
    parser.add_argument("--sampled_frames_num", type=int, default=7200)
    parser.add_argument("--output_dir", type=str, default="video_needle_haystack/data/haystack_vicuna_embeddings")
    parser.add_argument("--pooling_size", type=int, default=0)
    parser.add_argument("--add_newline_token", action="store_true")
    args = parser.parse_args()
    main(args)
