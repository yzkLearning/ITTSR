### evaluation
python main.py --save_dir ./eval/RRSSRD/TTSR \
               --reset True \
               --log_file_name eval.log \
               --eval True \
               --eval_save_results True \
               --num_workers 0 \
               --dataset RRSSRD \
               --dataset_dir ./RRSSRD/ \
               --model_path ./TTSR.pt