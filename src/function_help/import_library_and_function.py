import pandas as pd
import sys
import json
import numpy as np
import argparse
from sklearn.decomposition import NMF
import re, os
import joblib
import pickle
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report, roc_auc_score, make_scorer, auc, roc_curve
import shutil
import random
from collections import Counter
import glob
import math
from pathlib import Path
import time
from numpy import log,dot,exp,shape
# import matplotlib.pyplot as plt
# from IPython.display import clear_output

# how to initialize SNMF: W ~ K and H ~ X
def init_NMF(X, rank, init_mode):
    class CustomNMF(NMF):

        def init_nmf(self, X):
            X = self._validate_data(
                X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32], reset=False
            )

            self._check_params(X)
            
            # initialize W and H
            W = None
            H = None
            update_H = True
            W, H = self._check_w_h(X, W, H, update_H)

            return W, H

    nmf = CustomNMF(n_components=rank, init=init_mode, random_state=42)
    W, H = nmf.init_nmf(X)
    
    return W, H

# how to initialize W ~ K
def init_W(X, rank):
    n_samples, n_features = X.shape
    avg = np.sqrt(X.mean() / rank)
    W = np.full((n_samples, rank), avg, dtype=X.dtype)
    return W