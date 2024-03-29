# LaGAT: Link-aware Graph Attention Network for Drug-Drug Interaction Prediction

## Install

```
conda create -n LaGAT python=3.7
source activate LaGAT  
git clone git@github.com:Azra3lzz/LaGAT.git
cd LaGAT  
pip install -r requirement.txt 
```

## Usage

```
python run.py
  -d : Choose which dataset to use, the default is kegg
  -n :  Select the number of neighbor samples, the default is 64
  -hop : Select the depth of neighbor sampling, default is 1
  -b : The value of batchsize for each epoch of training, the default is 1024
  -lr : The value of learning rate for each epoch of training, the default is 1e-2
  -nd : The value of node feature's dimension , the default is 64
  -lc : this parameter decide whether to use Layer-wise aggregation layer,the default is 1
  -c : this parameter decide which attention layer to be used,the default is 3(LaGAT layer)
  -r: this parameter determines whether to export the test results for visualization, the default is 0
  -K: this parameter determines whether to test the generalization ability of the model in the cold start scenario, the default is 0
  -head : this parameter determines the number of Multi-head attention, the default is 1; it only takes effect when the -c parameter value is 2 (GAT layer)
```

## Dataset

We currently only provide the KEGG dataset in [raw_data](https://github.com/Azra3lzz/LaGAT/tree/main/raw_data), and will provide the Drugbank dataset in the future.

Note that the distribution of positive samples in the KEGG dataset is unbalanced; the negative samples in the KEGG dataset are randomly generated by us based on the 1925 drugs it contains and are stored in the file [approved_example.txt](https://github.com/Azra3lzz/LaGAT/blob/main/raw_data/kegg/approved_example.txt), so the overall drug distribution is the same as the distribution of positive samples. different.We generate new negative samples according to the distribution of positive samples and put them in [approved_example_balanced.txt](https://github.com/Azra3lzz/LaGAT/blob/main/raw_data/kegg/approved_example_balanced.txt), so that the overall drug distribution of the KEGG dataset will not be changed.

The default sample file used in the code is [approved_example.txt](https://github.com/Azra3lzz/LaGAT/blob/main/raw_data/kegg/approved_example.txt). If you want to use [approved_example_balanced.txt](https://github.com/Azra3lzz/LaGAT/blob/main/raw_data/kegg/approved_example_balanced.txt), please modify the corresponding code.

## Training

The training adopts `5` rounds of early-stopping strategy, the maximum number of training rounds is set not to exceed `50`, and the regularization coefficient is fixed to `1e-7`; Note that by default we randomly divide the data into 5 folds and take 5-fold cross-validation to test our model. It is also possible to use the `-K `parameter to control the use of new division folds, each fold contains only drugs that appear only in this fold, to test the model's generalization ability to drugs that do not appear in the training set.

