from sklearn.model_selection import KFold

from feature_extractor import feature_extractor
from model import model_build, ModelType


if __name__ == "__main__":
    generes = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    ]

    split = KFold(n_splits=3, shuffle=True)

    model = model_build(ModelType.RFC, split)

    file_path = "./test/Казённый унитаз - Моча съела говно.mp3"
    with open(file_path, "r") as file:
        features = feature_extractor(file)

    pred = model.predict(features)
    prob = model.predict_proba(features)
    print(prob)

    res = {}
    for index, key in enumerate(generes):
        res[key] = prob[0][index]

    print(f"Model prediction: {pred}")
    print(f"Probability: {res}")
