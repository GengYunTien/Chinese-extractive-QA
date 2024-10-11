#!/usr/bin/sh
#SBATCH -A MST109178         
#SBATCH -J paragraphSelection        
#SBATCH -p ngs1gput          
#SBATCH -c 6         
#SBATCH --mem=90g
#SBATCH --gres=gpu:1
#SBATCH -o PS_out.log           
#SBATCH -e PS_err.log          
#SBATCH --mail-user=willytien88@gmail.com    
#SBATCH --mail-type=BEGIN,END,FAIL             

module load pkg/Anaconda3
source activate /home/u1307362/anaconda3/envs/gnn

python paragraphSelection.py \
--context_file '/staging/biology/u1307362/Chinese-extractive-QA/context.json' \
--train_file '/staging/biology/u1307362/Chinese-extractive-QA/train.json' \
--validation_file '/staging/biology/u1307362/Chinese-extractive-QA/valid.json' \
--model_name_or_path '/staging/biology/u1307362/huggingface/hfl--chinese-bert-wwm-ext' \
--output_dir '/staging/biology/u1307362/Chinese-extractive-QA/paragraphSelection' \
--seed 42 \
