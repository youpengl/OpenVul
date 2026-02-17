#!/bin/bash
#SBATCH --cpus-per-task=64
#SBATCH --partition=             
#SBATCH --qos=           
#SBATCH --nodes=1                        
#SBATCH --ntasks-per-node=1             
#SBATCH --gres=[4 A100 80g]       
#SBATCH --mem=480gb                   
#SBATCH --time=5-00:00:00             
#SBATCH --job-name=orpo    
#SBATCH --output=logs/orpo.out     
#SBATCH --error=logs/orpo.err      

source ~/.bashrc

cd "please input your local path here"

source .venv/bin/activate

bash orpo.sh