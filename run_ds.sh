MY_RUN_NAME=ds_v2

rm -rf runs/$MY_RUN_NAME
rm -rf encode_runs/$MY_RUN_NAME

deepspeed hf_train.py --deepspeed ds_config.json \
    --model_name_or_path intfloat/simlm-base-msmarco \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --train_path "/traindata/datasets/embeddings/citr/mgaiduk/v0/citr_v1_5m_train/1pos8randneg.jsonl" \
    --val_path "/traindata/datasets/embeddings/citr/mgaiduk/v0/citr_v1_5k_val/1pos8randneg.jsonl" \
    --q_max_len 512 \
    --p_max_len 512 \
    --train_n_passages 2 \
    --max_steps 10000 \
    --learning_rate 5e-6 \
    --warmup_steps 50 \
    --fp16 \
    --output_dir "runs/${MY_RUN_NAME}" \
    --save_total_limit 2 \
    --save_strategy steps \
    --save_steps 300 \
    --remove_unused_columns False \
    --report_to wandb "$@" \
    --dataloader_num_workers 2 \
    --dataloader_prefetch_factor 50 \
    --evaluation_strategy steps \
    --logging_steps 50 \
    --eval_steps 300
