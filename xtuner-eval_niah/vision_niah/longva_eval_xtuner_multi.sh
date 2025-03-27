rm -r ./tmp/tmp2
mkdir ./tmp/tmp2
export TRITON_CACHE_DIR="./tmp/tmp2"
export PYTHONPATH="./"

for MODEL_NAME in LongVA-7B
do
mkdir vision_niah/data/haystack_embeddings/$MODEL_NAME
mkdir vision_niah/data_multi/needle_embeddings/$MODEL_NAME

JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
NUM_GPUS=8


python vision_niah/produce_haystack_embedding.py \
    --model vision_niah/model_weights/$MODEL_NAME \
    --video_path vision_niah/data/haystack_videos/video_haystack.mkv \
    --output_dir vision_niah/data/haystack_embeddings/$MODEL_NAME \
    --sampled_frames_num 3000 \
    --pooling_size 2 \
    2>&1 | tee vision_niah/log/s1/eval_${MODEL_NAME}_$(date +"%Y%m%d_%H%M").log



python vision_niah/multi_produce_needle_embedding.py \
    --model vision_niah/model_weights/$MODEL_NAME \
    --needle_dataset vision_niah/data_multi/source_data/niah-coco-multihop-100.json \
    --output_dir vision_niah/data_multi/needle_embeddings/$MODEL_NAME \
    --pooling_size 2 \
    2>&1 | tee vision_niah/log/s2/eval_multihop_${MODEL_NAME}_$(date +"%Y%m%d_%H%M").log



torchrun --nproc-per-node=${NUM_GPUS} vision_niah/multi_eval_vision_niah.py \
    --model  vision_niah/model_weights/$MODEL_NAME \
    --needle_dataset vision_niah/data_multi/source_data/niah-coco-multihop-100.json \
    --needle_embedding_dir vision_niah/data_multi/needle_embeddings/$MODEL_NAME \
    --haystack_dir vision_niah/data/haystack_embeddings/$MODEL_NAME \
    --prompt_template qwen2 \
    --max_frame_num 3000 \
    --min_frame_num  500 \
    --frame_interval 500 \
    2>&1 | tee vision_niah/log/s3/eval_multihop_${MODEL_NAME}_$(date +"%Y%m%d_%H%M").log

done