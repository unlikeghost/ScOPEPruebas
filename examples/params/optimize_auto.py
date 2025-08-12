from scope.utils import ScOPEOptimizerAuto

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
        "sample_1": ["toxic harmful dangerous poison lethal", "mercury lead arsenic cyanide"], 
        "sample_2": ["safe harmless beneficial healthy natural", "water vitamin protein calcium"]
    }
    for _ in range(24)
]


optimizer = ScOPEOptimizerAuto(
    random_seed=42,
    n_trials=100,
    target_metric='auc_roc',
    study_name="test_optimization"
)

study = optimizer.optimize(x_validation, y_validation, kw_samples_validation)

best_model = optimizer.get_best_model()

optimizer.save_complete_analysis(top_n=100)