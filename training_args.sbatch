#!/usr/bin/bash

#SBATCH -t 1-00:00:00
#SBATCH -N 1 --gpus-per-node=V100:1
#SBATCH -p alvis

if [ -z "$batch_size" ]
then
    batch_size=32
fi
if [ -z "$seed" ]
then
    seed=0
fi
if [ -z "$alpha" ]
then
    alpha=0.1
fi
if [ -z "$epsilon" ]
then
    epsilon=0.01
fi
if [ -z "$sigma" ]
then
    sigma=[3,2]
fi
if [ -z "$task" ]
then
    task=2
fi

. load_modules.sh

python training_args.py -bs $batch_size -s $seed -a $alpha -e $epsilon -s $sigma -t $task
