set -x

export PROJECT_FILE_LOC="<PROJECT_FILE_LOC>"

ulimit -n 65535

# PROJECT_DIR="$(pwd)"
PROJECT_DIR="./verl"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
DATA_PATH="<DATA_PATH>"
experiment_name="ToolTrain-32B"
save_path="<save_path>/$experiment_name"

pip install -e ./verl
cd ./verl

pip install liger-kernel
pip install tensordict==0.6.2
pip install -r ./verl/requirements_sglang.txt
pip uninstall -y wandb
pip install wandb
pip install pre-commit
pip install datasets
pip install tiktoken
pip install libcst
pip install llama-index
pip install jsonlines

CUDA_LAUNCH_BLOCKING=1 python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='repo_search_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    data.max_prompt_length=12288 \
    data.max_response_length=20480 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="Qwen-32B-path" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.multi_turn.max_turns=10 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.rollout.name=sglang_async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='func_loc_rl' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$nnodes \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.rollout_data_dir=$save_path/rollout_data \
    trainer.validation_data_dir=$save_path/validation_data \
    trainer.default_local_dir=$save_path \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    reward_model.reward_manager=prime \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/repo_search_tool_config.yaml" \
    trainer.total_epochs=15 $@ $1 2>&1 | tee log.txt

