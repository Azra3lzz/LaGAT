# LaGAT: Link-aware Graph Attention Network for Drug-Drug Interaction Prediction

This is the code for our paper ``LaGAT: Link-aware Graph Attention Network for Drug-Drug Interaction Prediction'' (published in Bioinformatics'22).

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
  -head : this parameter determines the number of Multi-head attention, the default is 1; it only takes effect when the -c parameter value is 1 (GAT layer)
```

## Contributing

PRs accepted.

## License

MIT © Richard McRichface
