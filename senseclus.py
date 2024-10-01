import logging
import warnings
import time
import hdbscan
import numpy as np
import pandas as pd
import umap.umap_ as umap
from hdbscan import flat
from sklearn.base import BaseEstimator, ClassifierMixin

from utils import check_is_df, extract_categorical, extract_numerical, normalize_array, cluster_coverage, cluster_counts

logger = logging.getLogger("senseclus")
logger.setLevel(logging.ERROR)
sh = logging.StreamHandler()
sh.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
)
logger.addHandler(sh)


class SenseClus(BaseEstimator, ClassifierMixin):
    """SenseClus

    Creates UMAP embeddings and HDSCAN clusters from mixed data

    Parameters
    ----------
            random_state : int, default=None
                Random State for both UMAP and numpy.random.
                If set to None UMAP will run in Numba in multicore mode but
                results may vary between runs.
                Setting a seed may help to offset the stochastic nature of
                UMAP by setting it with fixed random seed.

            n_neighbors : int, default=30
                Level of neighbors for UMAP.
                Setting this higher will generate higher densities at the expense
                of requiring more computational complexity.

            min_samples : int, default=15
                Samples used for HDBSCAN.
                The larger this is set the more noise points get declared and the
                more restricted clusters become to only dense areas.

            min_cluster_size : int, default=100
                Minimum Cluster size for HDBSCAN.
                The minimum number of points from which a cluster needs to be
                formed.

            n_components : int, default=logarithm
                Number of components for UMAP.
                These are dimensions to reduce the data down to.
                Ideally, this needs to be a value that preserves all the information
                to form meaningful clusters. Default is the logarithm of total
                number of features.

            cluster_selection_method: str, default=eom
                The HDBSCAN selection method for how flat clusters are selected from
                the cluster hiearchy. Defaults to EOM or Excess of Mass

            umap_combine_method : str, default=intersection
                Method by which to combine embeddings spaces.
                Options include: intersection, union, contrast,
                intersection_union_mapper
                The latter combines both the intersection and union of
                the embeddings.
                See:
                https://umap-learn.readthedocs.io/en/latest/composing_models.html

            prediction_data: bool, default=False
                Whether to generate extra cached data for predicting labels or
                membership vectors few new unseen points later. If you wish to
                persist the clustering object for later re-use you probably want
                to set this to True.
                See:
                https://hdbscan.readthedocs.io/en/latest/soft_clustering.html

            verbose : bool, default=False
                Level of verbosity to print when fitting and predicting.
                Setting to False will only show Warnings that appear.
    """

    def __init__(
            self,
            random_state: int = None,
            n_neighbors: int = 30,
            min_samples: int = 15,
            min_cluster_size: int = 100,
            n_components: int = None,
            cluster_selection_method: str = "eom",
            umap_combine_method: str = "intersection",
            prediction_data: bool = False,
            verbose: bool = False,
            flat_clusters: int = None,
    ):

        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.n_components = n_components
        self.cluster_selection_method = cluster_selection_method
        self.umap_combine_method = umap_combine_method
        self.prediction_data = prediction_data
        self.flat_clusters = flat_clusters

        if verbose:
            logger.setLevel(logging.DEBUG)
            self.verbose = True
        else:
            logger.setLevel(logging.ERROR)
            self.verbose = False

            # supress deprecation warnings
            # see: https://stackoverflow.com/questions/54379418

            def noop(*args, **kargs):
                pass

            warnings.warn = noop

        if isinstance(random_state, int):
            np.random.seed(seed=random_state)
        else:
            logger.info("No random seed passed, running UMAP in Numba")

    def __repr__(self):
        return str(self.__dict__)

    def fit(self, df: pd.DataFrame) -> None:
        """Fit function for call UMAP and HDBSCAN

        Parameters
        ----------
            df : pandas DataFrame
                DataFrame object with named columns of categorical and numerics

        Returns
        -------
            Fitted: None
                Fitted UMAPs and HDBSCAN
        """

        check_is_df(df)

        if not isinstance(self.n_components, int):
            self.n_components = int(round(np.log(df.shape[1])))

        logger.info("Extracting categorical features")
        self.categorical_ = extract_categorical(df)

        logger.info("Extracting numerical features")
        self.numerical_ = extract_numerical(df)

        logger.info("Fitting categorical UMAP")
        self._fit_categorical()

        logger.info("Fitting numerical UMAP")
        self._fit_numerical()

        logger.info("Fitting intersection UMAP")
        self._fit_intersection_umap()

        logger.info("Mapping/Combining Embeddings")
        self._umap_embeddings()

        logger.info("Fitting HDBSCAN...")
        self._fit_hdbscan()

    def _fit_numerical(self):
        numerical_umap = umap.UMAP(
            metric="l2",
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            min_dist=0.0,
            random_state=self.random_state,
        ).fit(self.numerical_)
        self.numerical_umap_ = numerical_umap
        return self

    def _fit_categorical(self):
        categorical_umap = umap.UMAP(
            metric="dice",
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            min_dist=0.0,
            random_state=self.random_state,
        ).fit(self.categorical_)
        self.categorical_umap_ = categorical_umap
        return self

    def _fit_intersection_umap(self):
        intersection_umap = umap.UMAP(
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            min_dist=0.0,
        ).fit(self.numerical_)
        self.intersection_umap_ = intersection_umap
        return self

    def _umap_embeddings(self):

        if self.umap_combine_method == "intersection":
            self.mapper_embedding_, self.min_val, self.max_val = normalize_array(
                self.numerical_umap_.embedding_ *
                self.categorical_umap_.embedding_)

        elif self.umap_combine_method == "union":
            self.mapper_embedding_, self.min_val, self.max_val = normalize_array(
                self.numerical_umap_.embedding_ +
                self.categorical_umap_.embedding_)

        elif self.umap_combine_method == "contrast":
            self.mapper_embedding_, self.min_val, self.max_val = normalize_array(
                self.numerical_umap_.embedding_ -
                self.categorical_umap_.embedding_)

        elif self.umap_combine_method == "intersection_union_mapper":
            self.mapper_embedding_, self.min_val, self.max_val = normalize_array(
                self.intersection_umap_.embedding_ *
                (self.numerical_umap_.embedding_ +
                 self.categorical_umap_.embedding_))

        else:
            raise KeyError("Select valid   combination method")

        return self

    def _fit_hdbscan(self):
        if self.flat_clusters:

            flat_model = flat.HDBSCAN_flat(
                X=self.mapper_embedding_,
                cluster_selection_method=self.cluster_selection_method,
                n_clusters=self.flat_clusters,
                min_samples=self.min_samples,
                metric="euclidean",
            )

            self.hdbscan_ = flat_model
        else:
            hdb = hdbscan.HDBSCAN(
                min_samples=self.min_samples,
                min_cluster_size=self.min_cluster_size,
                cluster_selection_method=self.cluster_selection_method,
                prediction_data=self.prediction_data,
                gen_min_span_tree=True,
                metric="euclidean",
            ).fit(self.mapper_embedding_)
            self.hdbscan_ = hdb
        return self

    def score(self):
        """Returns the cluster assigned to each row.

        This is wrapper function for HDBSCAN. It outputs the cluster labels
        that HDBSCAN converged on.

        Parameters
        ----------
        None : None

        Returns
        -------
        labels : np.array([int])
        """
        return self.hdbscan_.labels_


def fit_SenseClus(df, params):
    """Fits a DenseClus and returns all relevant information
        df = data
        params = dict with the prameters for the DenseClus

        returns -------------
        embedding =  transformed data points
        clustered = boolean vector decides if  not noise
        result = data frame with the embedding a and LABELS
        DBCV = score
        coverage = notNoise/total-points



    """
    np.random.seed(params['SEED'])
    clf = SenseClus(
        random_state=params['SEED'],
        cluster_selection_method=params['cluster_selection_method'],
        min_samples=params['min_samples'],
        n_components=params['n_components'],
        min_cluster_size=params['min_cluster_size'],
        umap_combine_method=params['umap_combine_method'],
        n_neighbors= params['n_neighbors'],
        prediction_data=True

    )

    start = time.time()
    clf.fit(df)
    print('time fitting ', (time.time() - start) / 60)
    print(clf.n_components)
    embedding = clf.mapper_embedding_
    labels = clf.score()

    result = pd.DataFrame(clf.mapper_embedding_)
    result['LABELS'] = pd.Series(clf.score())
    print('clusters ', len(set(result['LABELS'])) - 1)

    lab_count = result['LABELS'].value_counts()
    lab_count.name = 'LABEL_COUNT'

    lab_normalized = result['LABELS'].value_counts(normalize=True)
    lab_normalized.name = 'LABEL_PROPORTION'
    print('ruido ', lab_normalized[-1])

    labels = pd.DataFrame(clf.score())
    labels.columns = ['LABELS']
    print(cluster_counts(labels, 'LABELS'))
    clustered, coverage = cluster_coverage(labels, 'LABELS')
    print(f"Coverage :{coverage}")
    DBCV = clf.hdbscan_.relative_validity_
    return embedding, clustered, result, DBCV, coverage, clf
