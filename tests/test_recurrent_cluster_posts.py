from omegaconf import OmegaConf, DictConfig

from kygs.scripts.recurrently_cluster_posts import recurrently_cluster_posts
from tests.test_cluster_posts import cfg, embedding_cfg, message_provider_cfg, message_handlers_cfg, hac_clustering_cfg, centroid_based_clustering_cfg, run_and_check_clustering_based_on_cfg


class TestRecurrentClusterPosts:
    def test_hac(
        self,
        cfg: DictConfig,
        embedding_cfg,
        message_provider_cfg,
        message_handlers_cfg,
        hac_clustering_cfg,
        monkeypatch
    ):
        # Add mocks
        cfg.embedding = embedding_cfg
        cfg.message_provider = message_provider_cfg
        cfg.message_handlers = message_handlers_cfg
        cfg.clustering = hac_clustering_cfg
        cfg.output.save_dendrogram = False

        # Run and check
        run_and_check_clustering_based_on_cfg(recurrently_cluster_posts, cfg)

    def test_centroid_based(
        self,
        cfg: DictConfig,
        embedding_cfg,
        message_provider_cfg,
        message_handlers_cfg,
        centroid_based_clustering_cfg,
        monkeypatch
    ):
        # Add mocks
        cfg.embedding = embedding_cfg
        cfg.message_provider = message_provider_cfg
        cfg.message_handlers = message_handlers_cfg
        cfg.clustering = centroid_based_clustering_cfg
        cfg.output.save_dendrogram = False

        # Run and check
        run_and_check_clustering_based_on_cfg(recurrently_cluster_posts, cfg)
