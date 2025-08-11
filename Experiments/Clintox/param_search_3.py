import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import deepchem as dc

from scope.utils import ScOPEOptimizerBayesian
from scope.utils.report_generation import make_report
from scope.utils.sample_generation import SampleGenerator


seed: int = 42
np.random.seed(seed)
random.seed(seed)

STUDY_NAME: str = 'Clintox'
TEST_SAMPLES:list = 20
TRIALS: int = 1_00
CVFOLDS: int = 3
TIMEOUT: int = 1_800


# TARGET_METRIC: dict = {
#     'f1': 0.25,
#     'f2': 0.1,
#     'mcc': 0.1,
#     'auc_roc': 0.3,
#     'auc_pr': 0.25
# }
TARGET_METRIC: str = 'auc_roc'

SMILES_COLUMN: str = 'smiles'
LABEL_COLUMN: str = 'fda_approved'


RESULTS_PATH: str = os.path.join('results')
ANALYSYS_RESULTS_PATH: str = os.path.join(RESULTS_PATH, str(TEST_SAMPLES), 'Optimization')
EVALUATION_RESULTS_PATH: str = os.path.join(RESULTS_PATH, str(TEST_SAMPLES), 'Evaluation')


tasks, datasets, _ = dc.molnet.load_clintox(splitter='stratified', reload=True, feature_field='smiles', featurizer=dc.feat.DummyFeaturizer())
train_dataset, valid_dataset, test_dataset = datasets

train_smiles = train_dataset.ids
train_labels = train_dataset.y

valid_smiles = valid_dataset.ids
valid_labels = valid_dataset.y

test_smiles = test_dataset.ids  
test_labels = test_dataset

search_dataset = dc.data.DiskDataset.merge([test_dataset, valid_dataset])


test_dataset = dc.data.DiskDataset.merge([train_dataset])


df_test = pd.DataFrame({
    "smiles": test_dataset.X,
    "fda_approved": test_dataset.y[:, 0],
    "ct_tox": test_dataset.y[:, 1]
})

df_search = pd.DataFrame({
    "smiles": search_dataset.X,
    "fda_approved": search_dataset.y[:, 0],
    "ct_tox": search_dataset.y[:, 1]
})


print(df_search.head())
print(df_test.head())
print("\nDistribución de clases test:")
print(df_test[LABEL_COLUMN].value_counts())

print("\nDistribución de clases search:")
print(df_search[LABEL_COLUMN].value_counts())



x_test = df_test[SMILES_COLUMN].values
y_test = df_test[LABEL_COLUMN].values

x_search = df_search[SMILES_COLUMN].values
y_search = df_search[LABEL_COLUMN].values

print(x_test.shape, y_test.shape, x_search.shape, y_search.shape)


search_generator = SampleGenerator(
    data=x_search,
    labels=y_search,
    seed=seed,
)

optimizer = ScOPEOptimizerBayesian(
    free_cpu=0,
    n_trials=TRIALS,
    random_seed=seed,
    target_metric=TARGET_METRIC,
    timeout=TIMEOUT,
    study_name=f'{STUDY_NAME}_Samples_{TEST_SAMPLES}',
    output_path=ANALYSYS_RESULTS_PATH,
    cv_folds=CVFOLDS,
    use_cache=False
)
    
all_x = []
all_y = []
all_kw = []

for x_search_i, y_search_i, search_kw_samples_i in search_generator.generate(num_samples=TEST_SAMPLES):
    all_x.append(x_search_i)
    all_y.append(y_search_i)
    all_kw.append(search_kw_samples_i)


study = optimizer.optimize(all_x, all_y, all_kw)

optimizer.save_complete_analysis(top_n=1000)

best_model = optimizer.get_best_model()

test_generator = SampleGenerator(
    data=x_test,
    labels=y_test,
    seed=seed,
)

all_y_true = []
all_y_predicted = []
all_y_probas = []

for x_test_i, y_test_i, test_kw_samples_i in test_generator.generate(num_samples=TEST_SAMPLES):
    
    
    softmax_probs = list(best_model(
        list_samples=x_test_i,
        list_kw_samples=test_kw_samples_i
    ))[0]['softmax']
    
    class_names = sorted(softmax_probs.keys())
    proba_values = [softmax_probs[cls] for cls in class_names]
    
    predicted_class_idx = np.argmax(proba_values)
                        
    all_y_true.append(
        y_test_i
    )
    
    all_y_predicted.append(
        predicted_class_idx
    )
    
    all_y_probas.append(
        proba_values
    )
    
results = make_report(
    y_true=all_y_true,
    y_pred=all_y_predicted,
    y_pred_proba=all_y_probas,
    save_path=EVALUATION_RESULTS_PATH
)

print(results)