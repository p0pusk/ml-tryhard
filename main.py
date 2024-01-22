from sklearn.model_selection import KFold

from feature_extractor import feature_extractor
from model import model_build, ModelType


if __name__ == "__main__":
    split = KFold(n_splits=3, shuffle=True)

    model = model_build(ModelType.KNN, split)

    file_path = "./Data/genres_original/rock/rock.00089.wav"
    with open(file_path, "r") as file:
        features = feature_extractor(file)

    pred = model.predict(features)
    prob = model.predict_proba(features)

    print(f"Model prediction: {pred}")
    print(f"Probability: {prob}")
