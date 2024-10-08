#!/bin/bash

## 获取试卷
#python build_DKT.py --max_len=200 --train_size=0.8 \
#    # 数据集相关设置
#    --dataset=assistment --num_concepts=116 \
#    # 网络相关设置
#    --epoch=20 --lr=0.002 --batch_size=64 --hidden_size=64 --num_layers=1 \
#    # 保存相关设置
#    --save_root=saved_models --save_model_file=model
    

python build_DKT.py --max_len=200 --train_size=0.8 \
    --dataset=static --num_concepts=80 \
    --epoch=20 --lr=0.002 --batch_size=64 --hidden_size=64 --num_layers=1 \
    --save_root=saved_models --save_model_file=model

