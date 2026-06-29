# test_retrain_pipeline.py
import numpy as np
from unittest.mock import MagicMock
from api import recommender

def setup_fake_artifacts():
    recommender._model = MagicMock()
    recommender._model.predict.return_value = np.array([[0.5]])
    recommender._model.evaluate.return_value = 0.5
    recommender._user2id = {1: 0, 2: 1, 3: 2}
    recommender._id2user = {0: 1, 1: 2, 2: 3}
    recommender._movie2id = {101: 0, 102: 1}


def test_register_returns_three_values(tmp_path, monkeypatch):
    setup_fake_artifacts()
    monkeypatch.setattr(recommender, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(recommender, "USER2ID_PATH", tmp_path / "user2id.pkl")
    monkeypatch.setattr(recommender, "ID2USER_PATH", tmp_path / "id2user.pkl")

    result = recommender.register_user("testuser")
    assert len(result) == 3, "register_user must return (raw_id, internal_id, is_new)"
    raw_id, internal_id, is_new = result
    assert is_new is True
    assert isinstance(raw_id, int)
    assert isinstance(internal_id, int)


def test_retrain_model_accepts_internal_id_payload():
    setup_fake_artifacts()
    payload = [{"internal_id": 0, "movie_id": 101, "score": 1.0}]
    result = recommender.retrain_model(payload)  # raises KeyError today if desynced
    assert result["epochs"] >= 0


def test_name_to_raw_id_reads_registry(tmp_path, monkeypatch):
    import json
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"neel": 6041}))
    monkeypatch.setattr(recommender, "REGISTRY_PATH", registry_path)

    raw_id = recommender._name_to_raw_id("neel")
    assert raw_id == 6041