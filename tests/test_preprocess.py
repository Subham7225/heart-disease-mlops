import pandas as pd

from src.preprocess import build_preprocessor, load_dataset

def test_build_preprocessor_has_transformers():
    preprocessor = build_preprocessor()
    assert len(preprocessor.transformers) == 1

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
