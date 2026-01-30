cat > /external/nfs/01-home/$USER/qwen-lora/slurm_startup.sh << 'EOF'
#!/bin/bash
set -e

cd /external/nfs/01-home/$USER/qwen-lora

GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}

python3 -V
nvidia-smi || true

torchrun --nproc_per_node=$GPUS_PER_NODE train_lora_sft.py \
  --train_files train/sft_train_agents_eng.jsonl train/sft_train_agents_rus.jsonl train/sft_train_agents_temp_03_big.jsonl \
  --output_dir output/qwen3-4b-lora-sft \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  2>&1 | tee logs/run_${SLURM_JOB_ID}_${SLURM_PROCID}.log
EOF

chmod +x /external/nfs/01-home/$USER/qwen-lora/slurm_startup.sh
