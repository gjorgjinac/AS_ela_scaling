import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import sys
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor

from N_ranking_utils import *

import argparse
from config import *

parser = argparse.ArgumentParser(description="A script that performs LOPO ranking within a same dataset used for training and testing")
parser.add_argument("--algorithms", type=str, help="A dash-separated list of algorithms")
parser.add_argument("--dimension", type=int, help="Problem dimension")
parser.add_argument("--train_benchmark", type=str, help="Name of the benchmark to train on")



args = parser.parse_args()
sample_count_dimension_factor=50
dimension = args.dimension
algorithm_portfolio=args.algorithms
model_name="rf"
do_discrete_ranking=False #True if args.do_discrete_ranking.lower()=='true' else False
do_balance = False# True if args.do_balance.lower()=='true' else False
algorithms=[f'{algorithm_portfolio.split("_")[0].upper()}{i}' for i in range(1,11)] if 'config' in algorithm_portfolio else algorithm_portfolio.split('_')


random=Benchmark(name='random', id_columns=['f'], algorithm_portfolio=algorithm_portfolio,do_discrete_ranking=do_discrete_ranking, dimension=dimension,sample_count_dimension_factor=sample_count_dimension_factor )
bbob=Benchmark(name='bbob', id_columns=['f'] , algorithm_portfolio=algorithm_portfolio,do_discrete_ranking=do_discrete_ranking, dimension=dimension,sample_count_dimension_factor=sample_count_dimension_factor)
affine=Benchmark(name='affine', id_columns=['f'] , algorithm_portfolio=algorithm_portfolio,do_discrete_ranking=do_discrete_ranking, dimension=dimension,sample_count_dimension_factor=sample_count_dimension_factor)
m4=Benchmark(name='m4', id_columns=['f'] , algorithm_portfolio=algorithm_portfolio,do_discrete_ranking=do_discrete_ranking, dimension=dimension,sample_count_dimension_factor=sample_count_dimension_factor)

all_benchmarks={b.name:b for b in [bbob,affine,random,m4]}


real_models={'rf':RFModel()}
dummy_models={'rf':DummyModel()}
model=real_models[model_name]
dummy_model=dummy_models[model_name]


train_benchmarks = [args.train_benchmark] if args.train_benchmark is not None and args.train_benchmark in all_benchmarks.keys() else all_benchmarks.keys()
print("Training on benchmarks")
print(train_benchmarks)


result_dir=f'{data_dir}/results_ELA_SCALING_ANALYSIS_FEATURE_GROUPS/{sample_count_dimension_factor}d_samples/{dimension}d_generalization_discrete_ranking_{do_discrete_ranking}/{algorithm_portfolio}'
os.makedirs(result_dir,exist_ok=True) 
for train_benchmark in train_benchmarks:
    all_scores=pd.DataFrame()
    for fold in range(0,10):
        for budget in [10,30,50]:
            for features_to_use in ['ela','ela_sy','ela+ela_sy']:  
                all_benchmark_data={k: all_benchmarks[k].get_all_data( fold, budget, balance= do_balance and k==train_benchmark)  for k in all_benchmarks.keys() }

                print('train on', train_benchmark)
                test_benchmarks=list(set(all_benchmarks.keys()).difference({train_benchmark}))

                X_train=all_benchmark_data[train_benchmark][features_to_use]
                X_tests=[all_benchmark_data[b][features_to_use] for b in test_benchmarks]

                y_train=all_benchmark_data[train_benchmark]['y']
                y_tests=[all_benchmark_data[b]['y'] for b in test_benchmarks]

                feature_groups=set([c.split('.')[0] for c in X_train.columns] + ['all'])
                print(X_train.shape)
                print(feature_groups)
                for feature_group in feature_groups:
                    result_base_file_name=f'{result_dir}/model_{model.name}_budget_{budget}_fold_{fold}_train_{train_benchmark}/{feature_group}'
                    os.makedirs(result_base_file_name,exist_ok=True) 
                    if feature_group!='all':
                        feature_group_columns=list(filter(lambda x: x.startswith(f'{feature_group}'), X_train.columns))
                    else:
                        feature_group_columns=X_train.columns

                    test_precisions=[all_benchmarks[b].precision for b in test_benchmarks]
                    print(feature_group)
                    print(feature_group_columns)
                    fs_to_use=list(filter(lambda x: "costs_runtime" not in x and "costs_fun_evals" not in x, X_train[feature_group_columns].columns))
                    print(fs_to_use)
                    if len(fs_to_use)==0:
                        continue
                    try:
                        fold_scores,trained_model,_, _=model.run(X_train[fs_to_use], y_train,test_benchmarks,  [XX[fs_to_use] for XX in X_tests], y_tests,result_base_file_name, feature_name=features_to_use, test_precisions=test_precisions, budget=budget, save_feature_importance=True)
                    except Exception as e:
                        print("EXCEPTION")
                        print(e)
                        continue
                    fold_scores=pd.DataFrame(fold_scores)
                    fold_scores['fold']=fold
                    fold_scores['budget']=budget
                    fold_scores['algorithms']=algorithm_portfolio
                    fold_scores['train_name']=train_benchmark
                    fold_scores['features']=features_to_use
                    fold_scores['model']=model.name
                    fold_scores['feature_group']=feature_group
                    all_scores=pd.concat([all_scores,fold_scores])




                y_train=all_benchmark_data[train_benchmark]['full_y'].dropna()
                y_tests=[all_benchmark_data[b]['full_y'].dropna() for b in test_benchmarks]

                X_train=  pd.DataFrame(np.random.rand(*(y_train.shape)))
                X_tests=[pd.DataFrame(np.random.rand(*(y.shape))) for y in y_tests]
                print("Dummy x train")
                print(X_train.shape)
                print(X_train)
                fold_scores, trained_model,_, _=dummy_model.run(X_train,y_train,test_benchmarks,   X_tests, y_tests,result_base_file_name, feature_name='dummy', test_precisions=test_precisions, budget=budget)
                fold_scores=pd.DataFrame(fold_scores)
                fold_scores['features']='dummy'
                fold_scores['model']=model.name
                print(fold_scores)
                fold_scores['train_name']=train_benchmark
                fold_scores['fold']=fold
                fold_scores['budget']=budget
                fold_scores['algorithms']=algorithm_portfolio
                all_scores=pd.concat([all_scores,fold_scores])


                
            all_scores.to_csv(f'{result_dir}/all_scores_{model.name}_{train_benchmark}.csv')
    all_scores.to_csv(f'{result_dir}/all_scores_{model.name}_{train_benchmark}.csv')
