unset ROCR_VISIBLE_DEVICES
export WANDB_MODE=offline
export -n ROCR_VISIBLE_DEVICES  # Ensure it's fully removed
ray start --head --port=6379 --dashboard-host=0.0.0.0
ray start --address=<head_node_ip>:6379