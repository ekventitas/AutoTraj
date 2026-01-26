# module load cuda/12.1.1
cd /hy-tmp/LLaMA-Factory

# 可选：设置 WandB 项目名和运行名
export WANDB_PROJECT="AutoTraj"
export WANDB_RUN_NAME="Qwen2.5-7B-instruct-SFT"

CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train /hy-tmp/LLaMA-Factory/examples/train_full/qwen_sft_autotraj.yaml