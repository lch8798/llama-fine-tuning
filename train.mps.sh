torchrun --nproc_per_node=1 --master_port=34321 train.py \
    --model_name_or_path ./models/7B \
    --data_path ./datasets/alpaca/ko.json \
    --output_dir ./models/7B-ko \
    --use_mps_device \
    --no_cuda \
    --xpu_backend "gloo" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'