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
if [ -z "$sigma" ]
then 
    sigma='3,3'
fi
if [ -z "$epsilon" ]
then 
    epsilon=0.01
fi
if [ -z "$delta" ]
then 
    delta=0.05
fi
if [ -z "$binary" ]
then 
    binary=0
fi
if [ -z "$nclassifiers" ]
then 
    nclassifiers=2
fi
if [ -z "$bound" ]
then 
    bound='germain'
fi
if [ -z "$prior" ]
then 
    prior=''
fi
if [ -z "$posterior" ]
then 
    posterior=''
fi

. load_modules.sh

python batch_bound_single.py -t $task -r $seed -a $alpha -s $sigma -e $epsilon -d $delta -b $binary -n $nclassifiers -B $bound -p $prior -P $posterior