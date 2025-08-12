import numpy as np

from scope.utils.report_generation import make_report
from scope.utils import ScOPEOptimizerAuto
from scope.utils.sample_generation import SampleGenerator


x_validation = [
    "molecule toxic heavy metal lead", "compound dangerous poison arsenic", 
    "chemical harmful mercury substance", "element toxic cadmium dangerous",
    "poison lethal cyanide compound", "toxic substance benzene harmful",
    "dangerous chemical formaldehyde", "harmful compound asbestos fiber",
    "toxic metal chromium dangerous", "poison substance strychnine lethal",
    "harmful chemical dioxin toxic", "dangerous compound pesticide toxic",
    
    "safe molecule water oxygen", "harmless compound sugar glucose",
    "beneficial substance vitamin C", "safe chemical sodium chloride",
    "harmless element calcium safe", "beneficial compound protein amino",
    "safe substance carbohydrate energy", "harmless chemical citric acid",
    "beneficial molecule antioxidant", "safe compound fiber cellulose",
    "harmless substance mineral zinc", "beneficial chemical enzyme natural"
]

y_validation = [0]*12 + [1]*12

kw_samples_validation = [
    {
        "0": ["toxic harmful dangerous poison lethal", "mercury lead arsenic cyanide"], 
        "1": ["safe harmless beneficial healthy natural", "water vitamin protein calcium"]
    }
    for _ in range(24)
]


optimizer = ScOPEOptimizerAuto(
    random_seed=42,
    n_trials=100,
    target_metric='log_loss',
    study_name="test_optimization"
)

study = optimizer.optimize(x_validation, y_validation, kw_samples_validation)

best_model = optimizer.get_best_model()

optimizer.save_complete_analysis(top_n=100)


all_y_true = []
all_y_predicted = []
all_y_probas = []


preds = best_model(
    x_validation,
    list_kw_samples=kw_samples_validation
)

from collections import OrderedDict

for predx in preds:
    prediction = predx['probas']
    
    sorted_dict = OrderedDict(sorted(prediction.items()))

    
    pred_key = max(sorted_dict, key=sorted_dict.get) 
    
    
    predicted_class = int(pred_key.replace("sample_", ""))
    
    
    all_y_predicted.append(
        predicted_class
    )
    
    all_y_probas.append(
        list(sorted_dict.values())
    )
    
results = make_report(
    y_true=y_validation,
    y_pred=all_y_predicted,
    y_pred_proba=all_y_probas,
)
    

print(results)