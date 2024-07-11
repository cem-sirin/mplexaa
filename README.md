# mplexaa
mplexaa is a Python package for computing the Multiplex Adamic-Adar score, a measure of node similarity in multiplex networks introduced by Aleta et al. (2020) [[Arxiv](https://arxiv.org/abs/2011.09126)]
. The package also provides a class for creating and manipulating multiplex networks.


## Installation
To install mplexaa, run the following command:
```bash
pip install mplexaa
```

## Usage
The `MultiplexGraphDataset` class was first designed to read datasets from Manlio De Domenico's [Datasets Released for Reproducibility](https://manliodedomenico.com/data.php).

```python
from mplexaa import MultiplexGraphDataset, multiplex_adamic_adar

# Load the dataset
ds = MultiplexGraphDataset('csa')
maa = multiplex_adamic_adar(ds)
```

The file path for the datasets must be as follows:
```
project/
│-- datasets/
│   │-- csa/
│   │   │-- Dataset/
│   │   │   │-- edges.csv
│   │   │   │-- layers.csv
│   │-- ...
|-- main.py
```

An example of the project setup can be found the this [repository](https://github.com/cem-sirin/mplexaa).