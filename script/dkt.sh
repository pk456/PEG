#bin/bash
# 数据准备:执行dkt.sh

# 获取试卷
python build_DKT.py --max_len=200 --train_size=0.8 \
    # 数据集相关设置
    --dataset=c_filter2 --num_concepts=116 \
    # 网络相关设置
    --epoch=20 --lr=0.002 --batch_size=64 --hidden_size=64 --num_layers=1 \
    # 保存相关设置
    --save_root=saved_models --save_model_file=model
