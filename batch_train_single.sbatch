#!/usr/bin/bash

#SBATCH -t 1-00:00:00
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -p alvis 

if [ -z "$task" ]
then
    task=2
fi
if [ -z "$seed" ]
then 
    seed=69105
fi
if [ -z "$alpha" ]
then 
    alpha=0.0
fi
if [ -z "$delta" ]
then 
    delta=0.05
fi
if [ -z "$binary" ]
then 
    binary=0
fi
if [ -z "$architecture" ]
then 
    architecture='lenet'
fi
if [ -z "$image_size" ]
then 
    image_size=32
fi
if [ -z "$batch_size" ]
then 
    batch_size=128
fi
. load_modules.sh

python batch_train_single.py -t $task -r $seed -a $alpha -d $delta -b $binary -A $architecture -I $image_size -B $batch_size