
# Tail-GNN: Tail-Node Graph Neural Networks 
We provide the implementaion of Tail-GNN model.

The repository is organised as follows:
- dataset/: contains 3 benchmark datasets: squirrel, actor, cs-citation. Note: The larger cs-citation dataset is zipped, and must be unzipped first. Make sure the input files are all under dataset/cs-citation/ after unzipping. As all datasets will be processed on the fly, cs-citation may take 5+ mins to process before training will start. 
- models/: contains our model. 
- layers/: contains component layers for our model.  
- utils/: contains functions for data-processing, evaluation metrics, etc.
- link_prediction/: sub-directory to run the link prediction task.


## Requirements
To install required packages
- pip3 install -r requirements.txt

## Running experiments

### Tail node classification:
- python3 main.py --dataset=squirrel

For larger datasets such as as cs-citation, please use the sparse version:
- python3 main_sp.py --dataset=cs-citation


### Link prediction:
- cd link_prediction/
- python3 main.py --dataset=squirrel 

For larger datasets:
- python3 main_sp.py --dataset=cs-citation
