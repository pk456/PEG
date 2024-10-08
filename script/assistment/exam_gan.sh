#!/bin/bash
## 数据集搭建
#python examGAN_data.py \
#    # 数据集相关设置
#    --dataset=assistment --all_num_questions=15161 --num_concepts=116 \
#    # DKT网络相关设置
#    --load_model=./saved_models/assistment/model19 --hidden_size=64 --num_layers=1 \
#    # 班级试题相关设置
#    --num_students=50 --num_class=100 \
#    # 试题相关设置
#    --mean=70 --std=15 --num_questions=100 --num_init_papers=1000 \

## 获取试卷
#python main2.py --method=gan \
#    # 数据集相关设置
#    --dataset=assistment --all_num_questions=15161 --num_concepts=116 --train_data_file=gan/train_data.pkl \
#    # gan网络设置
#    --epoch=200 --batch_size=32 --random_dim=100 --lr=0.001 --gan_save_path=./saved_models/assistment/exam_gan \
#    # DKT网络相关设置
#    --load_model=./saved_models/assistment/model19 --hidden_size=64 --num_layers=1 \
#    # 班级试题相关设置
#    --num_students=50 \
#    # 试题相关设置
#    --mean=70 --std=15 --num_questions=100 --num_init_papers=1000 \
#    --save_paper=True --save_paper_path=./papers/



python examGAN_data.py \
    --dataset=assistment --all_num_questions=15161 --num_concepts=116 \
    --load_model=./saved_models/assistment/model19 --hidden_size=64 --num_layers=1 \
    --num_students=50 --num_class=100 \
    --mean=70 --std=15 --num_questions=100 --num_init_papers=1000 \



python main2.py --method=gan \
    --dataset=assistment --all_num_questions=15161 --num_concepts=116 --train_data_file=gan/train_data.pkl \
    --epoch=200 --batch_size=32 --random_dim=100 --lr=0.001 --gan_save_path=./saved_models/assistment/exam_gan \
    --load_model=./saved_models/assistment/model19 --hidden_size=64 --num_layers=1 \
    --num_students=50 \
    --mean=70 --std=15 --num_questions=100 --num_init_papers=1000 \
    --save_paper=True --save_paper_path=./papers/

python main2.py --method=t_gan --twin=True --num_papers=20\
    --dataset=assistment --all_num_questions=15161 --num_concepts=116 --train_data_file=gan/train_data.pkl \
    --epoch=200 --batch_size=32 --random_dim=100 --lr=0.001 --gan_save_path=./saved_models/assistment/exam_gan \
    --load_model=./saved_models/assistment/model19 --hidden_size=64 --num_layers=1 \
    --num_students=50 \
    --mean=70 --std=15 --num_questions=100 --num_init_papers=1000 \
    --save_paper=True --save_paper_path=./papers/
