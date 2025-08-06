#!/bin/bash
set -x
pip install ./verl
export EXPERIMENT_NAME=SFT-32B
pip uninstall -y wandb
pip install wandb
pip install liger-kernel
pip install tensordict==0.6.2

save_path="<save_path>"

mkdir -p $save_path

# Shift the arguments so $@ refers to the rest
# shift 2
train_batch_size=$((32 * nnodes))

torchrun --standalone --nnodes=$nnodes --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files="<train_files>" \
    data.val_files="<val_files>" \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.micro_batch_size=1 \
    data.micro_batch_size_per_gpu=1 \
    data.truncation=left \
    data.train_batch_size=$train_batch_size \
    data.max_length=16384 \
    model.partial_pretrain=Qwen-32B-path \
    trainer.default_local_dir=$save_path \
    trainer.project_name=swe-multiturn-sft \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=8 \
    use_remove_padding=true \
    model.enable_gradient_checkpointing=True \
    model.use_liger=True \
    2>&1 | tee $save_path/$EXPERIMENT_NAME.log
