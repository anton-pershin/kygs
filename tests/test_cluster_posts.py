import datetime
from collections import defaultdict

import numpy as np
import pytest
from omegaconf import OmegaConf, DictConfig
import hydra

from kygs.utils.typing import NDArrayFloat
from kygs.message_handler import MessageHandler
from kygs.message_provider import Message, MessageProvider
from kygs.scripts.cluster_posts import cluster_posts


MESSAGES = [
    {
        "text": "This is a test of clustering methods",
        "embedding": np.array([1.0, 1.0, 0.0]),  # coord 1 = test, coord 2 = clustering, coord 3 = cakes
        "true_label": "0", 
    },
    {
        "text": "This is a test targeting clustering methods",
        "embedding": np.array([1.01, 0.99, 0.0]),  # coord 1 = test, coord 2 = clustering, coord 3 = cakes
        "true_label": "0", 
    },
    {
        "text": "Test for clustering methods",
        "embedding": np.array([0.99, 1.01, 0.0]),  # coord 1 = test, coord 2 = clustering, coord 3 = cakes
        "true_label": "0", 
    },
    {
        "text": "This is a cake",
        "embedding": np.array([0.0, 0.0, 1.0]),  # coord 1 = test, coord 2 = clustering, coord 3 = cakes
        "true_label": "1", 
    },
    {
        "text": "Definitely a cake",
        "embedding": np.array([0.0, 0.0, 1.01]),  # coord 1 = test, coord 2 = clustering, coord 3 = cakes
        "true_label": "1", 
    },
    {
        "text": "Everyone talks about cakes",
        "embedding": np.array([0.0, 0.0, 0.99]),  # coord 1 = test, coord 2 = clustering, coord 3 = cakes
        "true_label": "1", 
    },
]

MESSAGE_HANDLER_DICT_STORAGE = {}


class MockEmbedding:
    def predict(self, text_sequences: list[str]) -> NDArrayFloat:
        embeddings = np.array([self._lookup_embedding_in_messages_by_text(t) for t in text_sequences])
        return embeddings

    @staticmethod
    def _lookup_embedding_in_messages_by_text(text: str) -> NDArrayFloat:
        global MESSAGES

        for m in MESSAGES:
            if text == m["text"]:
                return m["embedding"]

        raise ValueError(f"There is no text 'text' in MESSAGES")


class MessageDictSaver(MessageHandler):
    def handle(self, messages: list[Message], **kwargs) -> None:
        global MESSAGE_HANDLER_DICT_STORAGE
        
        clusters = defaultdict(set)
        for m in messages:
            if (m.label is not None) and (m.label != -1):
                clusters[m.label].add(m.text)

        MESSAGE_HANDLER_DICT_STORAGE["clusters"] = [cluster_set for label, cluster_set in clusters.items()]


@pytest.fixture
def cfg():
    with hydra.initialize(
        version_base="1.3",
        config_path="../config",
        job_name="test_app"
    ):
        default_cfg = hydra.compose(config_name="config_cluster_posts")
    
    return default_cfg


@pytest.fixture
def embedding_cfg():
    return {
        "_target_": "tests.test_cluster_posts.MockEmbedding",
    }


@pytest.fixture
def message_handlers_cfg():
    return [
        {
            "_target_": "tests.test_cluster_posts.MessageDictSaver",
        }
    ]


@pytest.fixture
def message_provider_cfg():
    return {
        "_target_": "tests.test_cluster_posts.create_message_dict_provider",
    }


@pytest.fixture
def hac_clustering_cfg(embedding_cfg):
    return {
        "_target_": "kygs.clustering.hac.HacBasedTextClustering",
        "_recursive_": True,
        "text_embedding_model": embedding_cfg,
        "distance_threshold": 0.1,
        "cluster_collection": {
            "_target_": "kygs.clustering.hac.FullEmbeddingClusterListCollection",
        },
        "linkage": "average",
    }


@pytest.fixture
def centroid_based_clustering_cfg(embedding_cfg):
    return {
        "_target_": "kygs.clustering.centroid_based.CentroidBasedTextClustering",
        "_recursive_": True,
        "text_embedding_model": embedding_cfg,
        "distance_threshold": 0.1,
        "cluster_collection": {
            "_target_": "kygs.clustering.centroid_based.CentroidBasedClusterListCollection",
        },
    }


def create_message_dict_provider() -> MessageProvider:
    global MESSAGES

    message_collection: list[Message] = []
    for raw_m in MESSAGES:
        m = Message(
            text=raw_m["text"],
            time=datetime.datetime.now(),
            author="test",
            label=None,
            true_label=raw_m["true_label"],
        )
        message_collection.append(m)

    return MessageProvider(messages=message_collection)


@pytest.fixture(autouse=True)
def clear_message_handler_dict_storage() -> None:
    global MESSAGE_HANDLER_DICT_STORAGE
    MESSAGE_HANDLER_DICT_STORAGE.clear()


class TestClusterPosts:
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
        run_and_check_clustering_based_on_cfg(cluster_posts, cfg)

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
        run_and_check_clustering_based_on_cfg(cluster_posts, cfg)


def run_and_check_clustering_based_on_cfg(cluster_func, patched_cfg: DictConfig):
    global MESSAGES
    global MESSAGE_HANDLER_DICT_STORAGE

    # Run the function being tested
    cluster_posts(patched_cfg)   

    true_cluster_1 = set(m["text"] for m in MESSAGES if m["true_label"] == "0")
    true_cluster_2 = set(m["text"] for m in MESSAGES if m["true_label"] == "1")
    
    # There must be only two clusters with no unclustered messages
    assert len(MESSAGE_HANDLER_DICT_STORAGE["clusters"]) == 2

    # Each of the clusters should be identical to the groundtruth
    assert (MESSAGE_HANDLER_DICT_STORAGE["clusters"][0] == true_cluster_1 and MESSAGE_HANDLER_DICT_STORAGE["clusters"][1] == true_cluster_2) or (MESSAGE_HANDLER_DICT_STORAGE["clusters"][0] == true_cluster_2 and MESSAGE_HANDLER_DICT_STORAGE["clusters"][1] == true_cluster_1)


