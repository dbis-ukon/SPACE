# SPACE: Cardinality Estimation for Path Queries Using Cardinality-Aware Sequence-based Learning

SPACE (Sequence-based Path Pattern Cardinality Estimator) is a deep-learning based approach designed to estimate the cardinalities of path patterns. Our model employs a recurrent neural network (RNN) architecture to accurately predict the cardinalities of path patterns. The proposed method, SPACE, operates in two main steps:

## 1.To collect the path patterns and computing cardinality vectors 

To collect path patterns and compute cardinality vectors, please refer to the instructions in the README file located in the 'Generate Training Data' directory.

## 2. Train the model and estimate cardinalities for path patterns

- Use the `run_estimation.py` script in 'Cardinality Estimation Model' directory to train the estimator and obtain estimated cardinalities for path patterns.


## Injecting Estimated Cardinalities into Graphflow


- Follow the instruction in README file located in the Graphflow directory.

- The original work of [Graphflow] ((https://github.com/colinsongf/graphflow) has been adapted in our paper.
