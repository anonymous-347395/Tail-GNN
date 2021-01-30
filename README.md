
# Tail-GNN
we provide the implementaion of Tail-GNN model:
Tail-GNN: Tail-Node Graph Neural Networks

The repository is organised as follows:
- dataset/: contains 3 benchmark datasets: squirrel, actor, cs-citation. Extract dataset before use
- models/: contains our models 
- layers/: contains component layers for models  
- utils/: contains functions for data-processing, metrics
- link_prediction/: sub repository to run the link prediction task


## Requirements
To install required packages
- pip3 install -r requirements.txt

## Running experiments

### Tail node classification:
- python3 main.py --dataset=squirrel

For large dataset as cs-citation, may need to use the sparse version:
- python3 main_sp.py --dataset=cs-citation


### Link prediction:
- cd link_prediction/
- python3 main.py --dataset=squirrel 

For large dataset:
- python3 main_sp.py --dataset=cs-citation
