#!/bin/bash

conda config --add channels https://conda.anaconda.org/gurobi
conda install -y coinbonmin coincbc glpk ipopt -c conda-forge
conda install -y gurobi -c gurobi

pip install -r requirements.txt
conda clean -a -y && rm -rf ~/.cache/pip

# grbgetkey bd7a2e9a-6303-11ec-9e9d-0242ac120002 -q
# grbgetkey 5cd711d8-64da-11ec-bac4-0242ac120002 -q
# grbgetkey c3fdf17e-6fd4-11ec-bb75-0242ac170002 -q