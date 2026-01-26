# 设置环境变量，让 Ray/vLLM/PyTorch 用 /hy-tmp 作为临时目录
export TMPDIR=/hy-tmp/tmp
export TEMP=/hy-tmp/tmp
export TMP=/hy-tmp/tmp

# 创建目录
mkdir -p $TMPDIR


export PYTHONPATH=/hy-tmp/Tool-Star/Tool_Star_RL/src/verl:$PYTHONPATH
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU



bash /hy-tmp/Tool-Star/Tool_Star_RL/scripts/train/train.sh \
    --train_batch_size 12 \
    --ppo_mini_batch_size 6 \
    --rollout_n 3 \
    --apply_chat True \
    --prompt_template_name re_search_template_sys \
    --actor_model_path /hy-tmp/AutoTraj/models/Qwen2.5-7B-Instruct \
    --project_name AutoTraj \
    --experiment_name AutoTraj_GRPO_7B_WO \
    --nnodes 1 \
    --n_gpus_per_node 4 \
    --save_freq 60 \
    --test_freq 60 \
    --total_epochs 1 \
    --wandb_api_key d2069356704ee1b8ce4f1f5f101c2d43ab9fd70d \
    --save_path /hy-tmp/AutoTraj/models/Qwen2.5-7B-Instruct-WO-GRPO \
    --train_files /hy-tmp/Tool-Star/Tool_Star_RL/mix_grpo/grpo_mix_train_shuffle.parquet \
    --test_files /hy-tmp/Tool-Star/Tool_Star_RL/mix_grpo/grpo_mix_test.parquet
    #--resume_from_checkpoint \
        #/hy-tmp/AutoTraj/models/Qwen2.5-7B-Instruct-GRPO-FULL/global_step_480 \