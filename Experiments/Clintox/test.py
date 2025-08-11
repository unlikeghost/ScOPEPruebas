import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import deepchem as dc

from scope import ScOPE
from scope.utils.report_generation import make_report
from scope.utils.sample_generation import SampleGenerator


seed: int = 42
np.random.seed(seed)
random.seed(seed)


SMILES_COLUMN: str = 'smiles'
LABEL_COLUMN: str = 'fda_approved'

tasks, datasets, _ = dc.molnet.load_clintox(splitter='stratified', reload=True, feature_field='smiles', featurizer=dc.feat.DummyFeaturizer())

train_dataset, valid_dataset, test_dataset = datasets

# valid_smiles = valid_dataset.ids
# valid_labels = valid_dataset.y

# test_smiles = test_dataset.ids  
# test_labels = test_dataset

all = dc.data.DiskDataset.merge([test_dataset, valid_dataset, train_dataset])

smiles = all.X
labels = all.y

all = pd.DataFrame({
    "smiles": smiles,
    "fda_approved": labels[:, 0],
    "ct_tox": labels[:, 1]
})

print(all.head())
print(all[LABEL_COLUMN].value_counts())


X = all[SMILES_COLUMN].values
Y = all[LABEL_COLUMN].values

search_generator = SampleGenerator(
    data=X,
    labels=Y,
    seed=seed,
)

model = ScOPE(
    compressor_names=[
        'bz2',
        'gzip',
        'zlib',
        'zstd'
    ]
    compression_metrics=[
        'ncd',
        'cdm',
        'nrc',
        'clm',
    ]
    use_symmetric_matrix=True,
    use_best_sigma=True,
    use_softmax=True,
    ensemble_strategy='voting',
)