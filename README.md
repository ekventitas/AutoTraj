# AutoTraj: Quick Start

This document provides a complete step-by-step guide for running Supervised Fine-Tuning (SFT), Reinforcement Learning (RL), and TIR Evaluation in AutoTraj.

## 🏷️ SFT Stage

### 1.1 Environment Setup
Install dependencies required for the SFT stage:

```bash
pip install -r for_sft_requirements.txt
```

The SFT stage is based on LLaMA-Factory. If you have not installed it yet, please follow the steps below (skip if already installed):

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[metrics]"
```

### 1.2 Supervised Fine-Tuning (SFT)

**Step 1: Prepare Dataset**  
Place the dataset at:  `LLaMA-Factory/data/sft_high_quality_trajectory_masked.json`  

Register the dataset in:  `LLaMA-Factory/data/dataset_info.json`

**Step 2: Configure Training**  
Complete the dataset path and training configuration in:  
`LLaMA-Factory/examples/train_full/qwen_sft_autotraj.yaml`

**Step 3: Run SFT Training**  
```bash
cd LLaMA-Factory
bash examples/train_full/train_sft.sh
```

### 1.3 Reward Model Training

**Step 1: Prepare Dataset**  
Place the reward model dataset at:  `LLaMA-Factory/data/rm_compare_pairs_masked.json`  

Register the dataset in:  `LLaMA-Factory/data/dataset_info.json`

**Step 2: Configure Reward Model**  
Edit the configuration file:  `LLaMA-Factory/examples/train_lora/qwen_lora_reward.yaml`

**Step 3: Train Reward Model**  
```bash
cd LLaMA-Factory
bash examples/train_lora/train_rm.sh
```

## 🚀 RL Stage

### 2.1 Environment Setup
Install dependencies required for RL training:

```bash
pip install -r grpo_requirements.txt
```

### 2.2 RL Training with Tool-Star and VERL
The RL training framework is built on VERL and Tool-Star.

**Step 1: Configure Training Script**  
Edit the following script and complete all required path and parameter settings:  
`Tool-Star/Tool_Star_RL/scripts/train/run_tool_star.sh`

**Step 2: Start Reward Model Server**  
Before starting RL training, launch the reward model service:

```bash
bash Tool-Star/Tool_Star_RL/src/verl/verl/utils/reward_score/rm_server.sh
```

**Step 3: Start RL Training**  
```bash
bash run_tool_star.sh
```

### 2.3 Core Code Reference
- **Rollout implementation**: `src/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py`
- **Reward calculation**: `src/verl/verl/utils/reward_score`

### 2.4 Convert RL Checkpoint to Hugging Face Format
Merge RL weights and export the model in Hugging Face format:

```bash
python Tool-Star/Tool_Star_RL/model_merger.py \
    --local_dir /{your_checkpoint_path}/global_step_{your_RL_step}/actor/
```

## 🛠️ TIR Evaluation

### 3.1 Environment Setup
Install inference dependencies:

```bash
pip install -r inference_requirements.txt
```

### 3.2 Retriever Serving Deployment
We deploy a Wikipedia retriever service based on FlashRAG and FastAPI.

**Step 1: Prepare Resources**  
Download:
- Pre-indexed Wikipedia
- Wikipedia corpus
- Corresponding retriever models

**Step 2: Configure Retriever Service**  
Edit the configuration file:  `evaluation/search/serving_config.yaml`  

Fill in:
- Retriever model path
- Index path
- Corpus path
- Available GPU IDs

**Step 3: Start Retriever Service**  
```bash
cd Tool-Star/evaluation/search
python host_wiki.py \
    --config serving_config.yaml \
    --num_retriever {num_retriever} \
    --port {port}
```

### 3.3 Inference
Run trajectory generation:

```bash
bash AutoTraj/trajectory_generate/start_generate.sh
```

### 3.4 Evaluation
Run trajectory evaluation:

```bash
bash AutoTraj/trajectory_generate/start_calculate.sh
```

## 📝 Notes
Ensure all dataset paths are correctly configured before training.
