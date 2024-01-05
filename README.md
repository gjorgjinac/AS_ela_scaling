# Impact of Scaling Objective Function Values in ELA Feature Calculation on Algorithm Selection Cross-Benchmark Generalizability

This repository contains the code for evaluating the impact of scaling objective function values before ELA feature calculation on the generalizability of an Algorithm Selection (AS) model across different benchmarks.

The "data" folder contains the ELA features calculated with and without scaling of the objective function values before ELA feature calculation. It also contains the performance data for the 4 algorithms on all 4 benchmarks.

The ela_feature_calculation.R script contains the code for calculating the ELA features.

The train_model.py contains the code for training and evaluating the AS model. It expects the following input arguments:
--algorithms (A dash-separated list of algorithms to include in the algorithm portfolio. Must match the order of the algorithms in the file name where the algorithm performance is saved)
--dimension (The dimension of the problem, 3 for our experiments. ELA features for this problem dimension must be calculated beforehand for all benchmarks and saved in the file "data/{benchmark}_{dimension}d.csv" or "data/{benchmark}_{dimension}d_scaled.csv"
--train_benchmark (Name of the benchmark to train the AS model on. Can be one of "bbob"/"random"/"affine"/"m4". The testing is done on the remaining benchmarks)

The remaining notebooks contain the code for generating the visualizations.

The "figures_results" folder contains the results for all tested dimensions (3 and 10).

Please note that in code we refer to the ZIGZAG benchmark listed in the paper as "m4"
