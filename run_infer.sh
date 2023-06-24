
# export CUDA_VISIBLE_DEVICES=4,5,6,7


SCRIPT=infer.py

deepspeed --num_nodes 1 --num_gpus 2  $SCRIPT
