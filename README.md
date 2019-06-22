## pcluster

A partial clustering wrapper for Mapper.






## Setup

### Dependencies

#### [Python 3.6+](https://www.python.org/)

#### Required Python Packages
* [numpy](www.numpy.org)
* [networkx](networkx.github.io)
* [nilearn](nilearn.github.io)

_For a full list of packages, see [`requirements.txt`](https://github.com/calebgeniesse/pcluster/blob/master/requirements.txt)._


### Install using pip

Assuming you have the required dependencies, you should be able to install using pip.
```bash
pip install git+https://github.com/calebgeniesse/pcluster.git
```

Alternatively, you can also clone the repository and build from source. 
```bash
git clone git@github.com:calebgeniesse/pcluster.git
cd pcluster

pip install -r requirements.txt
pip install -e .
```






## Usage

See below for a few examples of how `pcluster` can be used with `kmapper`. For more detailed usage, check out the [examples](https://www.github.com/calebgeniesse/pcluster/tree/master/examples/).


### The `PartialCluster` object

This package was initially developed around the idea of a `PartialCluster` class. The `PartialCluster` object wraps around a user-specified clustering algorithm, and can be pre-fit to the entire dataset before partial clustering on subsets of the data. To predict the cluster labels of a subset of data points, the pre-fit cluster labels are indexed based on a distance metric, and then used as the clustering labels.

Here, we will walk through the simple usage of the `PartialCluster`.

```python
from pcluster import PartialCluster
```

Here, we will use `nilearn` to fetch some example data from the Haxby dataset.
```python
import numpy as np 

# create some data
X = np.random.random(100, 2)
```

Next, we need to setup the `PartialCluster` model. 
```python
# initize the model
pclusterer = PartialCluster(
	clusterer=HDBSCAN(),
    verbose=3
)
```

Then simply fit the model to your data.
```python
pclusterer.precompute_fit(X)
```

Now, you can predict the labels of any subset of the data.
```python
labels = pclusterer.fit_predict(X[:10])
```