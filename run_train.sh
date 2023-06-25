
CFG=configs/ds_config_pp.json
SCRIPT=train_ds.py

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
deepspeed $SCRIPT --config $CFG

## 如果是多机的话，就加上一个这样的hostfile，里面写上哪几台服务器，并且各台里面有多少个gpu，
## 服务器用ssh的服务器名来表示，比如可以用ssh node1访问的服务器上面有8卡，在hostfile里面就应该有这么一行: node1 slots=8
# deepspeed --hostfile ./hostfile $SCRIPT --config $CFG

