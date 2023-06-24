
CFG=configs/ds_config_pp.json
SCRIPT=train_ds_pp.py
SCRIPT=train_ds.py
#SCRIPT=train_ds_tmp.py

## zero
# CFG=configs/ds_config_zero.json
# SCRIPT=train_ds_zero.py
# deepspeed $SCRIPT --deepspeed --deepspeed_config $CFG

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

## 如果是多机的话，就加上一个这样的hostfile，里面写上哪几台服务器，并且各台里面有多少个gpu，
## 服务器用ssh的服务器名来表示，比如可以用ssh gpu_1访问的服务器上面有8卡，在hostfile里面就应该有这么一行: gpu_1 slots=8
# deepspeed --hostfile ./hostfile $SCRIPT --config $CFG

## 单机的话，就不用加这个hostfile了
# deepspeed $SCRIPT --deepspeed --deepspeed_config $CFG

# 0.9.0要求initialize()函数里面指定cfg的话，就不能再使用--deepspeed_config了
deepspeed $SCRIPT --config $CFG

