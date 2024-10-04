#bin/bash
# 数据准备:执行dkt.sh

# 获取试卷
python main2.py --method=pga --epoch=100 \
    # 数据集相关设置
    --dataset=c_filter2 --all_num_questions=15161 --num_concepts=116 \
    # DKT网络相关设置
    --load_model=./saved_models/c_filter2/model19 --hidden_size=64 --num_layers=1 \
    # 班级试题相关设置
    --num_students=50 \
    # 试题相关设置
    --mean=70 --std=15 --num_questions=100 --num_init_papers=1000 \
    --save_paper=True --save_paper_path=./papers/ \
    # 生成相关设置
    --crossover_rate=0.8 ----mutation_rate=0.003
