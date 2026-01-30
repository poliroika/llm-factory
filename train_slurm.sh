cat > /external/nfs/01-home/$USER/qwen-lora/train_slurm.sh << EOF
#!/bin/bash
#SBATCH --job-name=qwen3-4b-lora
#SBATCH --partition=xgx
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --output=/external/nfs/01-home/$USER/qwen-lora/logs/%j.out
#SBATCH --error=/external/nfs/01-home/$USER/qwen-lora/logs/%j.err
#SBATCH --time=24:00:00

srun --container-image="/external/nfs/01-home/$USER/qwen-lora/images/CONTAINER.sqsh" \\
     --container-mounts="/external/nfs/:/external/nfs" \\
     --export="PYTHONPATH=/external/nfs/01-home/$USER/qwen-lora" \\
     --container-workdir="/external/nfs/01-home/$USER/qwen-lora" \\
     bash /external/nfs/01-home/$USER/qwen-lora/slurm_startup.sh
EOF

chmod +x /external/nfs/01-home/$USER/qwen-lora/train_slurm.sh
