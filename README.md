# Batch jobs

* [batch_queue_bounds.ipynb](batch_queue_bounds.ipynb) queues jobs to the server based on trained networks. Each checkpoint from the trained networks is submitted as its own job and the results are stored in pkl files containing dataframes with a single row. It only computes the various error terms but does not compute the final bound. 

* [batch_inspect_bounds.ipynb](batch_inspect_bounds.ipynb) computes the bounds and plots the results.


