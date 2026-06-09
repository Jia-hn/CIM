TIMESTAMP=$(date +%m%d-%H%M%S)
PROJECT_NAME=coco6k_internvl2_8b_grpo
SAVE_FREQ=6

python -m reward.reward_server &

cd rl_internvl
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=../data/SC-Captioner-data/llamafactory_json/train_coco6k_sft.parquet \
    data.val_files=../data/SC-Captioner-data/llamafactory_json/train_coco6k_sft.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=3500 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=OpenGVLab/InternVL2-8B \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +reward_model.ip=0.0.0.0 \
    +reward_model.port=3545 \
    +reward_model.api=cim \
    +reward_model.k=3 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=["console"] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name="$TIMESTAMP" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=2 \

cd ..
pids=()
for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i RANK=$i python -m eval.eval_internvl \
        --project_name $PROJECT_NAME \
        --experiment_name "$TIMESTAMP" \
        --save_freq $SAVE_FREQ &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
    wait $pid || fail=1
done
exit $fail
