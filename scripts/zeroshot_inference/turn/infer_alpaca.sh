#!/bin/bash

#SBATCH --job-name=infer_coh
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p new
#SBATCH --exclude hlt06

export mname='alpaca_hf_7B'
export model_dir=../llm_weights/${mname}

START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script started at: $START_TIME"

for lang in 'english' 'chinese' 'spanish' 'french' 'german' 'arabic' 'hindi' 'japanese' 'korean' 'russian'; do

    python zeroshot_inference/turn/run_alpaca.py \
        --output_folder outputs/zeroshot/${lang}_${mname} \
        --data_folder data/turn \
        --model ${model_dir} \
        --batch_size 1 \
        --lang ${lang} \
        --max_encode_len 2040

done

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script ended at: $END_TIME"
