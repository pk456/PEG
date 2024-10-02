#bin/bash
#This script is for PGA
python ./main2.py --method=pga --epoch=100 \
   --dataset=c_filter --num_students=2548 --num_concepts=122 --all_num_questions=17677 \
   --save_paper=True
