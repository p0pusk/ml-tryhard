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

from model import model_build


if __name__ == "__main__":
    split = KFold(n_splits=5, shuffle=True)
    rfc = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=10)

    model_build(rfc, split, "RandomForestClassifier")
