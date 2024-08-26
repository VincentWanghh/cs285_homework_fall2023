# # default setting
# for seed in $(seq 1 5); do
#     python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 \
#     --exp_name pendulum_default_s$seed \
#     -rtg --use_baseline -na \
#     --batch_size 5000 \
#     --seed $seed
# done
# Add some hyperparameters
for seed in $(seq 1 5); do
    python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 \
    --exp_name pendulum_discount_hyp_search_s$seed \
    -rtg --use_baseline -na \
    --discount 0.99 \
    --gae_lambda 0.99 \
    -s 256 \
    -l 3 \
    -lr 0.001 \
    -blr 0.002 \
    --batch_size 5000 \
    --baseline_gradient_steps 20 \
    --seed $seed
done

python cs285/scripts/run_hw2.py --env_name Humanoid-v4\
 --ep_len 1000 --discount 0.99 -n 1000 -l 3 -s 256 -b 50000 -lr 0.001\
 --baseline_gradient_steps 50\
 -na --use_reward_to_go --use_baseline --gae_lambda 0.97 \
 --exp_name humanoid --video_log_freq 5