#!/usr/bin/env bash
set -e

# =========================
# 基础环境变量
# =========================
export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========================
# 默认参数
# =========================
PROMPT_KEY=question
TRAIN_BATCH_SIZE=12
PPO_MINI_BATCH_SIZE=6
MAX_PROMPT_LENGTH=768
MAX_RESPONSE_LENGTH=2048
APPLY_CHAT=True
PROMPT_TEMPLATE_NAME=re_search_template_sys

ACTOR_MODEL_PATH=/hy-tmp/AutoTraj/models/Qwen2.5-7B-Instruct-SFT
REWARD_MANAGER=re_search
ROLLOUT_N=4
SEARCH_URL=http://localhost:1243

PROJECT_NAME=AutoTraj
EXPERIMENT_NAME=AutoTraj_GRPO_7B_FULL_RESUME

NNODES=1
N_GPUS_PER_NODE=3
SAVE_FREQ=80
TEST_FREQ=80
TOTAL_EPOCHS=1

WANDB_API_KEY=d2069356704ee1b8ce4f1f5f101c2d43ab9fd70d
SAVE_PATH="/hy-tmp/AutoTraj/models/Qwen2.5-7B-Instruct-GRPO-FULL"
TRAIN_FILES="/hy-tmp/Tool-Star/Tool_Star_RL/mix_grpo/grpo_train_remain.parquet"
TEST_FILES="/hy-tmp/Tool-Star/Tool_Star_RL/mix_grpo/grpo_mix_test.parquet"

# 🔴 新增：resume checkpoint
RESUME_FROM_CHECKPOINT="/hy-tmp/AutoTraj/models/Qwen2.5-7B-Instruct-GRPO-FULL/global_step_480"

# =========================
# 参数解析
# =========================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt_key) PROMPT_KEY="$2"; shift 2;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2;;
        --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2;;
        --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2;;
        --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2;;
        --apply_chat) APPLY_CHAT="$2"; shift 2;;
        --prompt_template_name) PROMPT_TEMPLATE_NAME="$2"; shift 2;;
        --actor_model_path) ACTOR_MODEL_PATH="$2"; shift 2;;
        --reward_manager) REWARD_MANAGER="$2"; shift 2;;
        --rollout_n) ROLLOUT_N="$2"; shift 2;;
        --search_url) SEARCH_URL="$2"; shift 2;;
        --project_name) PROJECT_NAME="$2"; shift 2;;
        --experiment_name) EXPERIMENT_NAME="$2"; shift 2;;
        --nnodes) NNODES="$2"; shift 2;;
        --n_gpus_per_node) N_GPUS_PER_NODE="$2"; shift 2;;
        --save_freq) SAVE_FREQ="$2"; shift 2;;
        --test_freq) TEST_FREQ="$2"; shift 2;;
        --total_epochs) TOTAL_EPOCHS="$2"; shift 2;;
        --wandb_api_key) WANDB_API_KEY="$2"; shift 2;;
        --save_path) SAVE_PATH="$2"; shift 2;;
        --train_files) TRAIN_FILES="$2"; shift 2;;
        --test_files) TEST_FILES="$2"; shift 2;;
        --resume_from_checkpoint) RESUME_FROM_CHECKPOINT="$2"; shift 2;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1;;
    esac
done

# =========================
# wandb
# =========================
if [ "$WANDB_API_KEY" != "None" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

# =========================
# 路径准备
# =========================
mkdir -p ${SAVE_PATH}
ROLLOUT_SAVE_PATH=${SAVE_PATH}/rollout
mkdir -p ${ROLLOUT_SAVE_PATH}

export PYTHONPATH=/hy-tmp/Tool-Star/Tool_Star_RL/src/verl:$PYTHONPATH
echo "[INFO] PYTHONPATH=$PYTHONPATH"

# =========================
# Resume / Init 逻辑（关键）
# =========================
MODEL_INIT_ARGS=""
RESUME_ARGS=""

if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    echo "[INFO] >>> Resuming from checkpoint: $RESUME_FROM_CHECKPOINT"
    RESUME_ARGS="+trainer.resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}"
else
    echo "[INFO] >>> Training from scratch / SFT init"
    MODEL_INIT_ARGS="actor_rollout_ref.model.path=${ACTOR_MODEL_PATH}"
fi

# =========================
# 启动训练
# =========================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${TEST_FILES}" \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.apply_chat=${APPLY_CHAT} \
    data.prompt_template_name=${PROMPT_TEMPLATE_NAME} \
    ${MODEL_INIT_ARGS} \
    ${RESUME_ARGS} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.peft.enable=True \
    +actor_rollout_ref.model.peft.type=lora \
    +actor_rollout_ref.model.peft.r=16 \
    +actor_rollout_ref.model.peft.alpha=32 \
    +actor_rollout_ref.model.peft.dropout=0.05 \
    +actor_rollout_ref.model.peft.target_modules='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
    +actor_rollout_ref.model.peft.freeze_base_model=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.sharding_strategy=SHARD_GRAD_OP \
    '+actor_rollout_ref.actor.fsdp_config.mixed_precision={param_dtype:bfloat16,reduce_dtype:bfloat16,buffer_dtype:bfloat16}' \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((2 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_with_search \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    +actor_rollout_ref.rollout.num_gpus=1 \
    +actor_rollout_ref.rollout.search_url=${SEARCH_URL} \
    +actor_rollout_ref.rollout.load_peft=True \
    +actor_rollout_ref.ref.load_peft=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((2 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward_model.reward_manager=${REWARD_MANAGER} \
    +reward_model.use_pretrained=False
    actor_rollout_ref.model.path=/hy-tmp/AutoTraj/models/Qwen2.5-7B-Instruct-GRPO-FULL/global_step_480/actor/huggingface \
    actor_rollout_ref.ref.model.path=/hy-tmp/AutoTraj/models/Qwen2.5-7B-Instruct-GRPO-FULL/global_step_480/actor/huggingface \
    actor_rollout_ref.rollout.model.path=/hy-tmp/AutoTraj/models/Qwen2.5-7B-Instruct-GRPO-FULL/global_step_480/actor/huggingface \
    reward_model.tokenizer_path=/hy-tmp/AutoTraj/models/Qwen2.5-7B-Instruct-GRPO-FULL/global_step_480/actor/huggingface \
    tokenizer_name_or_path=/hy-tmp/AutoTraj/models/Qwen2.5-7B-Instruct-GRPO-FULL/global_step_480/actor/huggingface \
    trainer.critic_warmup=0 \
    trainer.logger="[console, wandb]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${SAVE_PATH} \
    ++trainer.val_before_train=True \
    +trainer.rollout_save_path=${ROLLOUT_SAVE_PATH} \
    hydra.run.dir=${SAVE_PATH}/outputs | tee ${SAVE_PATH}/run.log
