# set group_num and group_id to process data in parallel
group_num=8
for group_id in $(seq 0 $((group_num - 1)))
do
    python data_process.py --data_path Data/val --save_root Data/processed/val --group_num $group_num --group_id $group_id --save_npz --check_size &
done

wait
