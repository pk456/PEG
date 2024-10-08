#!/bin/bash
# 数据准备:执行dkt.sh

## 获取试卷
#python main2.py --method=pdp --epoch=50 \
#    # 数据集相关设置
#    --dataset=assistment --all_num_questions=15161 --num_concepts=116 \
#    # DKT网络相关设置
#    --load_model=./saved_models/assistment/model19 --hidden_size=64 --num_layers=1 \
#    # 班级试题相关设置
#    --num_students=50 \
#    # 试题相关设置
#    --mean=70 --std=15 --num_questions=100 --num_init_papers=1000 \
#    --save_paper=True --save_paper_path=./papers/

python main2.py --method=pdp --epoch=50 \
    --dataset=static --all_num_questions=154 --num_concepts=80 \
    --load_model=./saved_models/static/model19 --hidden_size=64 --num_layers=1 \
    --num_students=50 \
    --mean=70 --std=15 --num_questions=100 --num_init_papers=1000 \
    --save_paper=True --save_paper_path=./papers/
