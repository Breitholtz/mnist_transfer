# Batch jobs

* [batch_queue_bounds.ipynb](batch_queue_bounds.ipynb) queues jobs to the server based on trained networks. Each checkpoint from the trained networks is submitted as its own job and the results are stored in pkl files containing dataframes with a single row. It only computes the various error terms but does not compute the final bound. 

* [batch_inspect_bounds.ipynb](batch_inspect_bounds.ipynb) computes the bounds and plots the results.

# Tasks

* The file [data/tasks.py](data/tasks.py) contains functions related to learning tasks including loading and preprocessing of data for experiments. 

# Bounds

* The file [bounds/bounds.py](bounds/bounds.py) has a function ```compute_bound_parts``` which computes the different terms which go into the Germain bounds for a pair of prior/posterior snapshots given as paths to .pkl files. 

* The file [batch_bound_single.py](batch_bound_single.py) implements the evaluation of a single snapshot through command line arguments. Calls ```compute_bound_parts```.

* The file [batch_bound_single.sbatch](batch_bound_single.sbatch) contains a Slurm script for queueing to the server, mirroring the arguments of ```batch_bound_single```. NOTE: The parameter sigma is supplied as a two-part string e.g., "3.3", not with a comma. 

* TODO: Should probably move ```calculate_germain_bound.py``` (and others) to ```bounds.py``` or vice versa. 
