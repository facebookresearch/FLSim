Model simulation framework simulates model training under FL setting, primarily
for ML engineers.

Model simulation framework aims to help ML engineers:
 - quickly train an existing model using FL and compare FL training to conventional training
 - quickly iterate to pick best model architecture and hyper-parameters for FL

At the same time, we want the model simulation framework to ensure:
 - quick training/iteration
 - efficient hyper-parameter tuning (Bayesian search)
 - developer environment familiar to ML engineers/researchers


Note that this is different from full-stack simulation (a.k.a. Systems Simulation)
in "papaya/simulation".


Components
 - utils
   Handy functionalities for simulating Federated Learning such as a class that
   can simulate event distribution for async FL
