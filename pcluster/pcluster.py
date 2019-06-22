import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator, TransformerMixin, ClusterMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import KDTree, BallTree

__all__ = [
    'PartialCluster',
    'SupervisedPartialCluster',
]




class PartialCluster(BaseEstimator, ClusterMixin):
    """Partial Cluster

    Parameters
    ----------
    clusterer : sklearn clustering object

    """
    def __init__(self,
            clusterer=None,
            min_samples=None,
            random_state=None,
            verbose=1,
        ):
        # setup clusterer
        self.clusterer = clusterer
        self.clusterer_params = clusterer.get_params()
        
        # over-ride min_samples if defined by clusterer
        self.min_samples = min_samples or self.clusterer_params.get('min_samples', 1) 

        # other parameters
        self.random_state = None
        self.verbose = verbose


    def get_clusterer(self, copy=False, **params):
        if copy is True:
            return clone(self.clusterer, **params)
        return self.clusterer

       
    def precompute_fit(self, X, y=None):
        """Precompute a clustering on X. If called before fit, the cluster
           labels will be assigned based on the precomputed labels.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.
        y : Ignored
            not used, present for API consistency by convention.

        Returns
        -------
        self : object
            returns instance of self

        """
        self._random_state = check_random_state(self.random_state)

        # check X, cache for later
        X = check_array(X)

        # fit the clusterer, cache for later
        self.precomputed_labels_ = self.clusterer.fit_predict(X, y=y)

        # now setup a tree for querying new points
        if hasattr(self.clusterer, 'cluster_centers_'):
            # print some info
            print("Setting up BallTree on cluster_centers_")
            print("*** cluster_centers_ has shape:", self.clusterer.cluster_centers_.shape)
            
            # setup BallTree using cluster centers
            self._knn = BallTree(self.clusterer.cluster_centers_, metric='euclidean', leaf_size=2)

        else:
            # print some info
            print("Setting up BallTree on X")
            print("*** X has shape:", X.shape)
            
            # use X if cluster_centers not available
            self._knn = BallTree(X, metric='euclidean', leaf_size=2)

        return self


    def fit(self, X, y=None):
        """Performs clustering on X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.
        y : Ignored
            not used, present for API consistency by convention.
        
        Returns
        -------
        self : object
            returns instance of self

        """
        self._random_state = check_random_state(self.random_state)

        # check X, cache for later
        X = check_array(X)
        
        # compute self.labels_
        if hasattr(self, 'precomputed_labels_'):
            # TODO: could potentially use k > 1 and weight by 1 / dist
            # find closest inds
            ind = self._knn.query(X, k=1, return_distance=False)[:, 0]

            # use precomputed labels
            self.labels_ = self.precomputed_labels_[ind]

        else:
            # just fit the clusterer, if not already computed
            self.labels_ = self.clusterer.fit_predict(X, y=y)

        # factorize the labels
        self.labels_, self.classes_ = pd.factorize(self.labels_) 

        # return instance
        return self


    def predict(self, X, y=None):
        """Returns cluster labels.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.
        y : Ignored
            not used, present for API consistency by convention.
        
        Returns
        -------
        labels : ndarray, shape (n_samples,)
            cluster labels

        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm        

        # TODO: might need to move logic here, i.e. 
        #       recompute nearest inds / labels for new X
        # return instance
        #if X.shape[0] != self.labels_.shape[0]:
        #    print("[warning] X.shape[0] != self.labels_.shape[0]")
        #    print ("*** Re-fitting on X...")
        #    self.fit(X)

        # check X, cache for later
        #X = check_array(X)
        
        # compute self.labels_
        #if hasattr(self, 'precomputed_labels_'):
        #    # TODO: could potentially use k > 1 and weight by 1 / dist
        #    # find closest inds
        #    ind = self._knn.query(X, k=1, return_distance=False)[:, 0]
        #
        #    # use precomputed labels
        #    labels = self.precomputed_labels_[ind]
        #
        #    # factorize the labels
        #    self.labels_, self.classes_ = pd.factorize(labels) 
        #
        # else:
        #    # just fit the clusterer, if not already computed
        #    self.labels_ = self.clusterer.fit_predict(X, y=y)

        # return labels
        return self.labels_

    
    def fit_predict(self, X, y=None):
        """Performs clustering on X and returns cluster labels.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.
        y : Ignored
            not used, present for API consistency by convention.
        Returns
        -------
        labels : ndarray, shape (n_samples,)
            cluster labels
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        self.fit(X, y=y)
        return self.predict(X)

    




class SupervisedPartialCluster(BaseEstimator, ClusterMixin):
    def __init__(self, k=1, min_samples=1):
        self.k = k
        self.min_samples = min_samples

    def precompute_fit(self, X, y=None):
        """Precompute a clustering on X. If called before fit, the cluster
           labels will be assigned based on the precomputed labels.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.
        y : Ignored
            not used, present for API consistency by convention.

        Returns
        -------
        self : object
            returns instance of self

        """
        # check X, cache for later
        X = check_array(X)

        # fit the clusterer, cache for later
        if y is None:
            self.precomputed_labels_ = np.arange(X.shape[0])
        else:
            self.precomputed_labels_ = y

        # use X if cluster_centers not available
        self._knn = BallTree(X, metric='euclidean', leaf_size=2)

        return self


    def fit(self, X, y=None):
        """Performs clustering on X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.
        y : Ignored
            not used, present for API consistency by convention.
        
        Returns
        -------
        self : object
            returns instance of self

        """
        # check X, cache for later
        X = check_array(X)
        
        # compute self.labels_
        if hasattr(self, 'precomputed_labels_'):
            # set labels
            self.labels_ = self.precomputed_labels_

        else: 
            # setup labels
            if y is None:
                self.labels_ = np.arange(X.shape[0])
            else:
                self.labels_ = y

            # setup knn
            self._knn = BallTree(X, metric='euclidean', leaf_size=2)

        # return instance
        return self


    def predict(self, X, y=None):
        """Returns cluster labels.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.
        y : Ignored
            not used, present for API consistency by convention.
        
        Returns
        -------
        labels : ndarray, shape (n_samples,)
            cluster labels

        """
        # check X, cache for later
        X = check_array(X)
                
        # TODO: could potentially use k > 1 and weight by 1 / dist
        # find closest inds
        ind = self._knn.query(X, k=self.k, return_distance=False)
        ind = ind[:, 0]

        # use precomputed labels
        labels = self.labels_[ind]

        # factorize the labels
        labels, self.classes_ = pd.factorize(labels) 

        # return labels
        return labels

    
    def fit_predict(self, X, y=None):
        """Performs clustering on X and returns cluster labels.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.
        y : Ignored
            not used, present for API consistency by convention.
        Returns
        -------
        labels : ndarray, shape (n_samples,)
            cluster labels
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        self.fit(X, y=y)
        return self.predict(X)

    

