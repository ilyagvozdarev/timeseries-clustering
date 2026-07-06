from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace, fields, is_dataclass

import numpy as np
from sklearn.manifold import TSNE

from clustering.n2d import n2d
from clustering.IDEC.IDEC import IDEC
from tslearn.clustering import KShape
from tslearn.metrics import cdist_normalized_cc
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from utils.io import read_config


class ModelConfig(ABC):
    @classmethod
    def from_config(cls, path: str | None = None):
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be decorated with @dataclass")
        config = cls()
        if path is None:
            return config
        overrides = read_config(path) or {}
        valid_keys = {f.name for f in fields(cls)}
        unknown = set(overrides) - valid_keys
        updated = valid_keys & set(overrides)
        if updated:
            print(f'{cls.__name__} updated: {updated}')
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        return replace(config, **overrides)

class Clusterizer(ABC):
    @abstractmethod
    def fit(self, x: np.ndarray): ...

    @abstractmethod
    def predict(self, x: np.ndarray): ...

    @abstractmethod
    def embed(self, x: np.ndarray): ...

    def embed_2d(self, x: np.ndarray) -> np.ndarray:
        if self.emb_dim == 2:
            return self.embed(x)
        embeds = self.embed(x)
        embeds_2d = TSNE(n_components=2, init='random', random_state=0).fit_transform(embeds)
        return embeds_2d
    
    def __call__(self, x):
        self.fit(x)
        return self.predict(x)

@dataclass
class N2DConfig(ModelConfig):
    epochs: int = 300
    umap_dim: int = 2
    umap_neighbors: int = 10
    umap_min_dist: float = 0.0
    umap_metric: str = 'euclidean'

class N2DClusterizer(Clusterizer):
    def __init__(self, n_clusters: int, cfg: N2DConfig):
        self.cfg = cfg
        self.n_clusters = n_clusters
        self.emb_dim = cfg.umap_dim

    def fit(self, x: np.ndarray):
        ae = n2d.AutoEncoder(x.shape[-1], self.n_clusters)
        manifold = n2d.UmapGMM(
            self.n_clusters, 
            umap_dim=self.cfg.umap_dim, umap_neighbors=self.cfg.umap_neighbors, 
            umap_min_distance=self.cfg.umap_min_dist, umap_metric=self.cfg.umap_metric
        )
        self.model = n2d.n2d(ae, manifold)
        self.model.fit(x, epochs=self.cfg.epochs, verbose=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def embed(self, x: np.ndarray) -> np.ndarray:
        self.model.predict(x)
        return self.model.hle   # umap_dim

@dataclass
class IDECConfig(ModelConfig):
    batch_size: int = 32
    maxiter: int = 300 
    pretrain_epochs: int = 100 
    gamma: float = 0.1          # clustering loss coefficient
    update_interval: int = 0    # one epoch
    tol: float = 0.00001        # stopping criterion - the proportion of objects whose cluster labels change 
                                # during one iteration from the previous iteration, below which training stops
    optimizer: str = 'adam'
    hidden_size: list = field(default_factory=lambda: [500, 500, 2000, 10])     

class IDECClusterizer(Clusterizer):
    def __init__(self, n_clusters: int, cfg: IDECConfig):
        self.cfg = cfg
        self.n_clusters = n_clusters
        self.emb_dim = cfg.hidden_size[-1]

    def fit(self, x: np.ndarray):
        update_interval = self.cfg.update_interval or int(x.shape[0] / self.cfg.batch_size)
        self.model = IDEC(dims=[x.shape[-1], *self.cfg.hidden_size], n_clusters=self.n_clusters)
        self.model.pretrain(x, batch_size=self.cfg.batch_size, epochs=self.cfg.pretrain_epochs, optimizer=self.cfg.optimizer)
        self.model.compile(loss=['kld', 'mse'], loss_weights=[self.cfg.gamma, 1], optimizer=self.cfg.optimizer)
        self.model.fit(
            x, y=None, 
            batch_size=self.cfg.batch_size, 
            tol=self.cfg.tol, maxiter=self.cfg.maxiter,
            update_interval=update_interval, ae_weights=None
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.y_pred

    def embed(self, x: np.ndarray) -> np.ndarray:
        return self.model.encoder.predict(x)    # hidden_size[-1] = latent_dim 

@dataclass
class KShapeConfig(ModelConfig):
    max_iter: int = 10      
    n_init: int = 1         # number of runs of k-Shape algorithm with different seeds for centroids, 
                            # final result is best result for n_init.
                            
class KShapeClusterizer(Clusterizer):
    def __init__(self, n_clusters: int, cfg: KShapeConfig):
        self.cfg = cfg
        self.n_clusters = n_clusters
        self.emb_dim = None

    def fit(self, x: np.ndarray):
        x = self._scale(x)
        self.model = KShape(
            self.n_clusters, max_iter=self.cfg.max_iter, 
            n_init=self.cfg.n_init, random_state=0, verbose=True
        ).fit(x)
        return self

    def _scale(self, x: np.ndarray) -> np.ndarray:
        # z-norm
        return TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(self._scale(x))

    def embed(self, x: np.ndarray) -> np.ndarray:
        return self.dists(x)
        
    def dists(self, x: np.ndarray) -> np.ndarray:
        # (n_samples, n_samples) SBD
        x = self._scale(x)
        norms = np.linalg.norm(x, axis=(1, 2))
        sim_matrix = cdist_normalized_cc(x, x, norms1=norms, norms2=norms, self_similarity=True)
        dist_matrix = 1. - sim_matrix
        # avoiding possible negative value due to numerical error
        dist_matrix = np.clip(dist_matrix, 0, None)
        # just in case, the diagonal should be exactly 0
        np.fill_diagonal(dist_matrix, 0.0)               
        return dist_matrix

    def embed_2d(self, x: np.ndarray) -> np.ndarray:
        dists = self.dists(x)
        embeds_2d = TSNE(
            n_components=2, metric='precomputed', init='random', random_state=0
        ).fit_transform(dists)
        return embeds_2d


CONFIGS = {
    'n2d': N2DConfig, 
    'idec': IDECConfig, 
    'kshape': KShapeConfig
}
CLUSTERIZERS = {
    'n2d': N2DClusterizer, 
    'idec': IDECClusterizer, 
    'kshape': KShapeClusterizer
}