
python split.py  
python target.py --train_seed 1  --gpu 0 &
python shadow.py --split_seed 1  --gpu 1 &
python shadow.py --split_seed 2  --gpu 1 &
python shadow.py --split_seed 3  --gpu 1 &
python shadow.py --split_seed 4  --gpu 1 &
python shadow.py --split_seed 5  --gpu 2 &
python shadow.py --split_seed 6  --gpu 2 &
python shadow.py --split_seed 7  --gpu 2 &
python shadow.py --split_seed 8  --gpu 2 &
python shadow.py --split_seed 9  --gpu 3 &
python shadow.py --split_seed 10 --gpu 3 &
python shadow.py --split_seed 11 --gpu 3 &
python shadow.py --split_seed 12 --gpu 3 &
python shadow.py --split_seed 13 --gpu 4 &
python shadow.py --split_seed 14 --gpu 4 &
python shadow.py --split_seed 15 --gpu 4 &
python shadow.py --split_seed 16 --gpu 4 &

