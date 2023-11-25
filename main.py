import numpy as np
import pandas as pd
from pandas.core.common import random_state
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from tempfile import TemporaryFile
from sklearn import preprocessing
import os
import math
import pickle
import random
import operator
import librosa
import librosa.display
import warnings
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from feature_extractor import feature_extractor

from model import model_build, ModelType


if __name__ == "__main__":
    split = KFold(n_splits=5, shuffle=True)

    model = model_build(ModelType.KNN, split)

    with open("./Data/genres_original/hiphop/hiphop.00003.wav", "r") as file:
        features = feature_extractor(file)

    pred = model.predict(features)
    prob = model.predict_proba(features)
    print(f"Model prediction: {pred}")
    print(f"Probability: {prob}")
