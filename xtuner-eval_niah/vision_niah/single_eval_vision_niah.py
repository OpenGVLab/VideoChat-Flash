import sys
import os
import argparse
import gc
import sys
import torch
from transformers import AutoTokenizer
from transformers import Qwen2ForCausalLM
from tqdm import tqdm
import glob
import numpy as np
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
import random
import json
from datasets import load_dataset


import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer, AutoConfig
from xtuner._lite.accelerate.dispatches import dispatch_modules
from mmengine.dist import infer_launcher, init_dist
from mmengine.runner import set_random_seed
from xtuner._lite.parallel import (split_for_sequence_parallel)
from xtuner._lite.parallel.setup import get_sp_group, setup_parallel

import torch.distributed as dist
from transformers.cache_utils import DynamicCache
import argparse


SEED = 24242424
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

prompt_templates = {
    "mistral": {
        "preprompt": "<s>[INST]",
        "postprompt": " [/INST]"
    },
    "vicuna": {
        "preprompt": "<s>A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER:",
        "postprompt": "ASSISTANT:"
    },
    "llama3": {
        "preprompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "postprompt": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "qwen2": {
        "preprompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
        "postprompt": "<|im_end|>\n<|im_start|>assistant\n",
    }, 
    "yi": {
        "preprompt": "<|im_start|>system\nAnswer the questions.<|im_end|>\n<|im_start|>user\n",
        "postprompt": "<|im_end|>\n<|im_start|>assistant\n",
    },
}



def safe_tokenize(tokenizer, text):
    tokenized = tokenizer.encode(text, return_tensors="pt")
    if tokenizer.bos_token != None and len(tokenized) > 0 and tokenized[0, 0] == tokenizer.bos_token_id:
        tokenized = tokenized[:, 1:]
    return tokenized

# answer = "more bet"
def eval_forward( model, input_embeds, answer_embeds, pad_id, answer_ids, tokenizer, sp_size, config, idx):
    world_size = dist.get_world_size(get_sp_group())
    rank = dist.get_rank(get_sp_group())
    
    prompt_length = input_embeds.shape[1]
    labels_length = answer_embeds.shape[1]
    
    input_embeds = torch.cat([input_embeds, answer_embeds], dim=1)
    pad_token_num = (sp_size * 2) - input_embeds.shape[1] % (sp_size * 2)
    pad_tensor = torch.tensor([pad_id] * pad_token_num).unsqueeze(0).unsqueeze(-1).expand(-1, -1, input_embeds.shape[-1]).to("cuda")
    input_embeds = torch.cat([input_embeds, pad_tensor], dim=1)
    position_ids = (
        torch.arange(input_embeds.shape[1]).unsqueeze(0).expand(input_embeds.shape[0], -1)
    ).to("cuda")
    seq_len_per_gpu = int(input_embeds.shape[1] // sp_size)
    # print("seq_len_per_gpu: ", seq_len_per_gpu)
    if rank == 0 :
        print("input_embeds: ", input_embeds.shape)
    
    assert input_embeds.shape[1] % sp_size == 0
    sp_group = get_sp_group()
    input_embeds = split_for_sequence_parallel(input_embeds, dim=1, sp_group=sp_group)  #原本这里是input_id, input_embeds也能用这个函数split吗？？
    position_ids = split_for_sequence_parallel(position_ids, dim=1, sp_group=sp_group)
    past_key_values = DynamicCache(config.num_hidden_layers)
    
    with torch.inference_mode():
        output = model(
            inputs_embeds=input_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = output[0]
        # print("rank: ", rank, "logits shape: ", logits.shape)
        pred = logits.argmax(dim=-1)
    
    dist.broadcast(pred, src=world_size - 1)    
    pred = pred[:, (prompt_length - 1)%seq_len_per_gpu : (prompt_length + labels_length - 1)%seq_len_per_gpu]
    # print("rank: ", rank, "pred shape: ", pred.shape)
    
    # check if the logits are correct, extract argmax id    # compare the predicted_ids with the labels
    predict_str = str(tokenizer.decode(pred.squeeze().tolist())).lower()
    answer_str = str(tokenizer.decode(answer_ids.to("cuda").squeeze().tolist())).lower()
    correct = (predict_str.replace("(","").replace(")","") == answer_str)
    if rank == 0 :
        print(
            " Idx:",
            idx,
            " Predicted:",
            tokenizer.decode(pred.squeeze().tolist()),
            " Answer:",
            tokenizer.decode(answer_ids.squeeze().tolist()),
            " Correct:",
            correct
        )
    return int(correct)


def load_haystack(args):
    haystack_embeddings = torch.load(f"{args.haystack_dir}/video_embeddings.pt").to(torch.bfloat16)
    # for file_path in tqdm(sorted(Path(args.haystack_dir).glob("*.pt"))[:args.max_frame_num], desc="Loading Haystack Embeddings...", disable=not accelerator.is_main_process):
    #     embeddings = torch.load(file_path, map_location="cuda").to(torch.bfloat16).unsqueeze(0)
    #     haystack_embeddings = embeddings if haystack_embeddings is None else torch.cat(
    #         [haystack_embeddings, embeddings], dim=0
    #     )
    return haystack_embeddings

def load_text_embeddings(str, tokenizer, model, replace_double_newline=False): 
    token_ids = safe_tokenize(tokenizer, str)
    def replace_double_newline_func(token_ids):
        # subsitute token id 271 to two 198]
        # for example:
        # from: tensor([[128000, 128006,   9125, 128007,    271,   2675,    527,    264,  11190, 4221,    323,  11376,  18328,     13]])
        # to: tensor([[128000, 128006,   9125, 128007,    198,    198,    2675,    527,    264,  11190, 4221,    323,  11376,  18328,     13]])
        # length will increase by number of 271
        double_newline_loc = (token_ids == 271).nonzero()[:, 1]
        double_newline_loc += torch.arange(len(double_newline_loc))
        if len(double_newline_loc) > 0:
            for loc in double_newline_loc:
                token_ids = torch.cat([token_ids[:, :loc], torch.tensor([[198, 198]]), token_ids[:, loc+1:]], dim=1)
        return token_ids
    if replace_double_newline:
        token_ids = replace_double_newline_func(token_ids)
    token_ids = token_ids.to("cuda")
    with torch.inference_mode():
        embeddings = model.model.embed_tokens(token_ids)
    return embeddings.to(torch.bfloat16)

def inference(args):
    
    dist_launcher = infer_launcher()
    init_dist(dist_launcher)
    set_random_seed(42)

    sp_size = dist.get_world_size()
    setup_parallel(sp_size, ring_size=sp_size)
    
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        model_max_length=sys.maxsize,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"rope_theta": args.rope_theta} if args.rope_theta is not None else {}

    config = AutoConfig.from_pretrained(args.model)
    model = Qwen2ForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        **kwargs,
    ).cuda()
    

    
    
    
    print("Preparing Haystack...")
    haystack_embeddings = load_haystack(args)
    assert len(haystack_embeddings) >= args.max_frame_num, "Haystack embeddings are not enough. Max frame {} is not found. Currently only {} frames.".format(args.max_frame_num, len(haystack_embeddings))
    haystack_embeddings = haystack_embeddings[:args.max_frame_num].to("cuda")
    prompt = prompt_templates[args.prompt_template]
    
    preprompt_embeddings = load_text_embeddings(prompt["preprompt"], tokenizer, model, args.replace_double_newline)
    postprompt_embeddings = load_text_embeddings(prompt["postprompt"], tokenizer, model, args.replace_double_newline)
    
    # needle_dataset = load_dataset(args.needle_dataset)["test"]
    
    with open(args.needle_dataset, 'r', encoding='utf-8') as file: 
        needle_dataset = json.load(file)
    
    answer_embedding_list = []
    answer_id_list = []
    needle_embedding_list = []
    question_embeding_list = []
    for index, instance in enumerate(needle_dataset):
        answer = instance["answer"]
        question = instance["question"]
        
        print("index",index,"\nquestion",question,"\nanswer",answer,"\n")
        needle_embedding_list.append(torch.load(args.needle_embedding_dir + f"/{index}.pt", map_location="cuda").to(torch.bfloat16).to("cuda"))
        answer_embedding_list.append(load_text_embeddings(answer, tokenizer, model))
        answer_id_list.append(safe_tokenize(tokenizer, answer))
        question_embeding_list.append(load_text_embeddings(question, tokenizer, model))
    


    print("Starting Evaluation...")
    model.eval()
    dispatch_modules(model)
    rank = dist.get_rank(get_sp_group())
    all_accuries = []
    for num_frames in tqdm(
        range(
            args.min_frame_num, args.max_frame_num + 1, args.frame_interval
        )
    ):
        for depth in np.arange(0, 1 + args.depth_interval, args.depth_interval):
            accuracies = []
            for idx, (question_embedding, needle_embedding, answer_embedding, answer_id) in enumerate(zip(question_embeding_list, needle_embedding_list, answer_embedding_list, answer_id_list)):
                query_frame_idx = int(depth * num_frames)
                # if rank==0:
                #     print(idx)
                
                haystack_embeddings_left = haystack_embeddings[:query_frame_idx].view(1, -1, haystack_embeddings.shape[-1])
                needle_embedding = needle_embedding.unsqueeze(0).view(1, -1, haystack_embeddings.shape[-1])
                haystack_embeddings_right = haystack_embeddings[query_frame_idx:num_frames].view(1, -1, haystack_embeddings.shape[-1])
                input_frames = torch.cat([haystack_embeddings_left,needle_embedding, haystack_embeddings_right], dim=1)
                
                # input_frames = torch.cat([haystack_embeddings[:query_frame_idx],needle_embedding.unsqueeze(0), haystack_embeddings[query_frame_idx:num_frames]], dim=0).view(-1, haystack_embeddings.shape[-1]).unsqueeze(0)
                input_emebds = torch.cat([preprompt_embeddings, input_frames,question_embedding, postprompt_embeddings], dim=1)
                
                # >>> eval forward start >>>
                correct = eval_forward(
                    model, input_emebds, answer_embedding, tokenizer.pad_token_id, answer_id, tokenizer, sp_size, config, idx
                )
                # <<< eval forward end <<<
                gc.collect()
                torch.cuda.empty_cache()
                accuracies.append(correct)
            if rank == 0:
                result = {
                    "Num. Frame": num_frames,
                    "Frame Depth": round(depth * 100, -1),
                    "Score": sum(accuracies) / len(accuracies),
                }
                print(result)
                all_accuries.append(result)
    if rank == 0:
        model_name = args.model.split("/")[-1]
        os.makedirs(f"{args.output_path}/{model_name}", exist_ok=True)
        # save all_accuries as json
        with open(f"{args.output_path}/{model_name}/all_accuracies.json", "w") as f:
            json.dump(all_accuries, f, indent=4)
        return all_accuries




def plot(args,  all_accuries):
    df = pd.DataFrame(all_accuries)
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["#F0496E", "#EBB839", "#9ad5b3"]
    )

    pivot_table = pd.pivot_table(
        df,
        values="Score",
        index=["Frame Depth", "Num. Frame"],
        aggfunc="mean",
    ).reset_index()  # This will aggregate
    pivot_table = pivot_table.pivot(
        index="Frame Depth", columns="Num. Frame", values="Score"
    )
    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    ax = sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        vmin=0,
        vmax=1,
        linecolor='white',
        linewidths=1.5, 
        cmap=cmap,
        cbar_kws={"label": "Score"},
    )
    
    # Set the color bar label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(14)
    cbar.ax.tick_params(labelsize=14)

    
    # Define the formatter function
    def thousands_formatter(x, pos):
        if x >= 1000:
            return f'{x/1000:.1f}K'
        return f'{x}'

    context_lengths = pivot_table.columns
    formatted_context_lengths = [thousands_formatter(x, None) for x in context_lengths]

    # More aesthetics
    plt.xlabel("Num. of Frames", fontsize=14)  # X-axis label
    plt.ylabel("Depth Percent", fontsize=14)  # Y-axis label
    plt.xticks(ticks=[i + 0.5 for i in range(len(context_lengths))], labels=formatted_context_lengths, rotation=45, fontsize=14)
    # plt.xticks(rotation=45, fontsize=14)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0, fontsize=14)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    # save
    model_name = args.model.split("/")[-1]

    plt.savefig(f"{args.output_path}/{model_name}/heatmap.png")
    # calculate average accuracy
    average_accuracy = df["Score"].mean()
    print(f"Average Accuracy: {average_accuracy}")
    # save as txt
    with open(f"{args.output_path}/{model_name}/avg_accuracy.txt", "w") as f:
        f.write(f"Average Accuracy: {average_accuracy}\n")
        
def main(args):
    if args.plot_only:
        # load all_accuracies from json
        model_name = args.model.split("/")[-1]
        with open(f"{args.output_path}/{model_name}/all_accuracies.json", "r") as f:
            all_accuracies = json.load(f)
        plot(args, all_accuracies)
    else:
        all_accuracies = inference(args)
        if dist.get_rank(get_sp_group()) == 0:
            plot(args, all_accuracies)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="output/LLaVA-NeXT-Video-7B-32K")
    args.add_argument("--max_frame_num", type=int, default=256)
    args.add_argument("--needle_dataset", type=str, default="lmms-lab/v_niah_needles")
    args.add_argument("--min_frame_num", type=int, default=20)
    args.add_argument("--frame_interval", type=int, default=20)
    args.add_argument("--output_path", type=str, default="vision_niah/niah_output_single")
    args.add_argument("--depth_interval", type=float, default=0.1)
    args.add_argument("--num_samples", type=int, default=1)
    args.add_argument("--rope_theta", type=float, default=None)
    args.add_argument("--haystack_dir", type=str, default="video_needle_haystack/data/haystack_embeddings")
    args.add_argument("--needle_embedding_dir", type=str, default="vision_niah/data/needle_embeddings")
    args.add_argument("--prompt_template", type=str)
    args.add_argument("--replace_double_newline", action="store_true")
    args.add_argument("--plot_only", action="store_true")
    
    main(args.parse_args())
