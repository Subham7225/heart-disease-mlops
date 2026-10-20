import pandas as pd

from src.preprocess import build_preprocessor, load_dataset


def test_build_preprocessor_has_transformers():
    preprocessor = build_preprocessor()
    names = [t[0] for t in preprocessor.transformers]
    assert names == ["continuous", "binary", "categorical"]

def test_load_dataset_reads_target(tmp_path):
    df = pd.DataFrame({
        "age": [60], "sex": [1], "cp": [2], "trestbps": [120], "chol": [240],
        "fbs": [0], "restecg": [1], "thalach": [150], "exang": [0], "oldpeak": [1.2],
        "slope": [1], "ca": [0], "thal": [2], "target": [1]
    })
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    bundle = load_dataset(path)
    assert bundle.X.shape == (1, 13)
    assert bundle.y.iloc[0] == 1


def test_preprocessor_imputes_and_encodes():
    df = pd.DataFrame({
        "age": [60, 45], "sex": [1, 0], "cp": [2, 4], "trestbps": [120, None], "chol": [240, 200],
        "fbs": [0, 1], "restecg": [1, 0], "thalach": [150, 170], "exang": [0, 1], "oldpeak": [1.2, 0.0],
        "slope": [1, 2], "ca": [0, None], "thal": [3, 7],
    })
    out = build_preprocessor().fit_transform(df)
    dense = out.toarray() if hasattr(out, "toarray") else out
    assert dense.shape[0] == 2
    assert dense.shape[1] > 13  # one-hot encoding expands categorical columns
    assert not pd.isna(dense).any()


def test_load_dataset_missing_target_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"age": [1]}).to_csv(path, index=False)
    try:
        load_dataset(path)
    except ValueError as exc:
        assert "target" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")
