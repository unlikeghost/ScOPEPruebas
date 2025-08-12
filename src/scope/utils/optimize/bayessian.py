import os
import warnings
import pandas as pd
from typing import List, Dict, Optional, Any, Union

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from scope.utils.optimize.base import ScOPEOptimizer
from scope.utils.optimize.params import ParameterSpace

warnings.filterwarnings('ignore')


class ScOPEOptimizerBayesian(ScOPEOptimizer):
    """Bayesian optimization for ScOPE models using Optuna."""
    
    def __init__(self, 
                 parameter_space: Optional[ParameterSpace] = None,
                 free_cpu: int = 0,
                 random_seed: int = 42,
                 cv_folds: int = 3,
                 study_name: str = "scope_optimization",
                 output_path: str = "./results",
                 n_trials: int = 50,
                 target_metric: Union[str, Dict[str, float]] = 'auc_roc',
                 use_cache: bool = True
                 ):
        """Initialize the Bayesian optimizer"""
        super().__init__(
            parameter_space=parameter_space, 
            free_cpu=free_cpu,
            random_seed=random_seed,
            cv_folds=cv_folds,
            study_name=study_name,
            output_path=output_path,
            n_trials=n_trials,
            target_metric=target_metric,
            use_cache=use_cache
        )
        
        os.makedirs(self.output_path, exist_ok=True)
    
    def optimize(self,
                X_validation: List[str],
                y_validation: List[str],
                kw_samples_validation: List[Dict[str, Any]]) -> optuna.Study:
        """Run Bayesian optimization"""
        
        print("=== BAYESIAN OPTIMIZATION SCOPE ===\n")
        print(f"Validation data: {len(X_validation)} samples")
        print(f"Classes: {sorted(set(y_validation))}")
        print(f"Trials: {self.n_trials}")
        print(f"CV Folds: {self.cv_folds}\n")
        
        self.validate_data(y_validation)
        self.validate_optimization_setup()
        
        self.print_target_metric_info()
        print()
        
        self.print_parameter_space()
        
        objective_func = self._create_objective_function(
            X_validation, y_validation, kw_samples_validation
        )
        
        direction = self.get_optimization_direction()
        
        # Configure Optuna study
        self.study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(
                seed=self.random_seed,
                n_startup_trials=50,
                n_ei_candidates=100,
                multivariate=False,
                group=False,
                constant_liar=True
            ),
            # pruner=MedianPruner(
            #     n_startup_trials=5,
            #     n_warmup_steps=1
            # ),
            study_name=f"{self.study_name}_{self.study_date}",
            storage=f"sqlite:///{self.output_path}/optuna_{self.study_name}.sqlite3",
            load_if_exists=True
        )
        
        # Execute optimization
        print("\nStarting optimization...")
        self.study.optimize(
            objective_func,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs
        )
        
        # Store best results
        self.best_params = self.study.best_params
        self.best_model = self.create_model_from_params(self.best_params)
        
        # Show results
        print("\n=== OPTIMIZATION RESULTS ===")
        print(f"Best score: {self.study.best_value:.4f}")
        
        # Analyze best model configuration
        compressor_names = self.best_params['compressor_names'].split(',')
        compression_metric_names = self.best_params['compression_metric_names'].split(',')
        
        # Determine if it's an ensemble
        is_ensemble = len(compressor_names) > 1 or len(compression_metric_names) > 1
        
        if is_ensemble:
            ensemble_strategy = self.best_params.get('ensemble_strategy', 'max')
            aggregation_strategy = self.best_params.get('agregation_strategy', None)
            print(f"Best model type: Ensemble ({ensemble_strategy} strategy)")
            print(f"Compressors: {compressor_names}")
            print(f"Compression metrics: {compression_metric_names}")
            print(f"Ensemble strategy: {ensemble_strategy}")
            if aggregation_strategy is not None:
                print(f"Aggregation strategy: {aggregation_strategy}")
        else:
            print(f"Best model type: Individual ({self.best_params.get('model_type', 'unknown')})")
            print(f"Compressor: {compressor_names[0]}")
            print(f"Compression metric: {compression_metric_names[0]}")
        
        cache_size = self._eval_cache_size_()
        total_trials = len(self.study.trials)

        cached_percent = 100 * cache_size / total_trials if total_trials > 0 else 0
        not_cached_percent = 100 - cached_percent
        
        print(f"Cached evaluations (unique parameter sets): {cache_size} ({cached_percent:.2f}%)")
        print(f"Non-cached evaluations (evaluations recomputed): {total_trials - cache_size} ({not_cached_percent:.2f}%)")

        self.print_best_configuration()
        
        return self.study

    def analyze_results(self):
        """Analyze optimization results"""
        
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        print("\n=== DETAILED ANALYSIS ===")
        
        # Basic statistics
        completed_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])
        
        print(f"Completed trials: {completed_trials}")
        print(f"Pruned trials: {pruned_trials}")
        print(f"Failed trials: {failed_trials}")
        
        # Ensemble vs Individual analysis
        df_results = self.get_trials_dataframe()
        if not df_results.empty:
            print(f"\nTotal unique parameter combinations evaluated: {len(df_results)}")
            
            # Analyze ensemble vs individual
            ensemble_trials = []
            individual_trials = []
            
            for idx, row in df_results.iterrows():
                compressor_names = str(row.get('params_compressor_names', '')).split(',')
                compression_metric_names = str(row.get('params_compression_metric_names', '')).split(',')
                
                is_ensemble = len(compressor_names) > 1 or len(compression_metric_names) > 1
                
                if is_ensemble:
                    ensemble_trials.append(row)
                else:
                    individual_trials.append(row)
            
            print(f"\nEnsemble vs Individual Analysis:")
            print(f"  Ensemble trials: {len(ensemble_trials)}")
            print(f"  Individual trials: {len(individual_trials)}")
            
            if ensemble_trials:
                ensemble_df = pd.DataFrame(ensemble_trials)
                print(f"  Best ensemble score: {ensemble_df['value'].max():.4f}")
            
            if individual_trials:
                individual_df = pd.DataFrame(individual_trials)
                print(f"  Best individual score: {individual_df['value'].max():.4f}")
        
        # Parameter importance
        try:
            importances = optuna.importance.get_param_importances(self.study)
            print("\nParameter importance:")
            print("-" * 40)
            
            # Separate parameters by category
            basic_params = {}
            ot_params = {}
            pd_params = {}
            ensemble_params = {}
            
            for param, importance in importances.items():
                if param.startswith('ot_'):
                    ot_params[param] = importance
                elif param.startswith('pd_'):
                    pd_params[param] = importance
                elif param in ['ensemble_strategy', 'agregation_strategy']:
                    ensemble_params[param] = importance
                else:
                    basic_params[param] = importance
            
            # Print basic parameters
            if basic_params:
                print("Basic Parameters:")
                for param, importance in sorted(basic_params.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {param}: {importance:.6f}")
                print()
            
            # Print ensemble parameters
            if ensemble_params:
                print("Ensemble Parameters:")
                for param, importance in sorted(ensemble_params.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {param}: {importance:.6f}")
                print()
            
            # Print ScOPE-OT parameters
            if ot_params:
                print("ScOPE-OT Parameters:")
                for param, importance in sorted(ot_params.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {param}: {importance:.6f}")
                print()
            
            # Print ScOPE-PD parameters
            if pd_params:
                print("ScOPE-PD Parameters:")
                for param, importance in sorted(pd_params.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {param}: {importance:.6f}")
                print()
            
            # Print overall ranking
            print("Overall Ranking (Top 10):")
            top_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
            for param, importance in top_params:
                print(f"  {param}: {importance:.6f}")
            
        except Exception as e:
            print(f"Could not calculate parameter importance: {e}")
        
        if not df_results.empty:
            print("\nTop 5 configurations:")
            columns_to_show = ['value', 'params_compressor_names', 'params_compression_metric_names', 
                              'params_model_type', 'params_use_best_sigma', 'params_use_symmetric_matrix', 
                              'params_ensemble_strategy', 'params_agregation_strategy']
            
            # Add model-specific columns if they exist
            model_specific_columns = []
            for col in df_results.columns:
                if col.startswith('params_ot_') or col.startswith('params_pd_'):
                    model_specific_columns.append(col)
            
            columns_to_show.extend(model_specific_columns)
            available_columns = [col for col in columns_to_show if col in df_results.columns]
            top_trials = df_results.nlargest(5, 'value')[available_columns]
            print(top_trials)
        
        return df_results
    
    def save_analysis_report(self, filename: Optional[str] = None):
        """Save detailed analysis report to text file"""
        
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        if filename is None:
            filename = f"{self.study_name}_analysis_{self.study_date}.txt"

        filepath = os.path.join(self.output_path, filename)
        
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SCOPE BAYESIAN OPTIMIZATION ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Study information
            f.write("STUDY INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Study name: {self.study_name}\n")
            f.write(f"Study date: {self.study_date}\n")
            f.write(f"Output path: {self.output_path}\n\n")
            
            # Study configuration
            f.write("OPTIMIZATION CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            
            if self.is_combined:
                f.write("Target metric: Combined\n")
                f.write("Metric weights:\n")
                for metric, weight in self.target_metric_weights.items():
                    f.write(f"  {metric}: {weight:.3f}\n")
            else:
                f.write(f"Target metric: {self.target_metric_name}\n")
            
            f.write(f"Optimization direction: {self.get_optimization_direction()}\n")
            f.write(f"Number of trials: {self.n_trials}\n")
            f.write(f"CV folds: {self.cv_folds}\n")
            f.write(f"Random seed: {self.random_seed}\n")
            f.write(f"Best score achieved: {self.study.best_value:.6f}\n\n")
            
            # Best model configuration
            compressor_names = self.best_params['compressor_names'].split(',')
            compression_metric_names = self.best_params['compression_metric_names'].split(',')
            is_ensemble = len(compressor_names) > 1 or len(compression_metric_names) > 1
            
            if is_ensemble:
                ensemble_strategy = self.best_params.get('ensemble_strategy', 'max')
                aggregation_strategy = self.best_params.get('agregation_strategy', None)
                f.write(f"Best model type: Ensemble ({ensemble_strategy} strategy)\n")
                f.write(f"Compressors: {compressor_names}\n")
                f.write(f"Compression metrics: {compression_metric_names}\n")
                f.write(f"Ensemble strategy: {ensemble_strategy}\n")
                if aggregation_strategy is not None:
                    f.write(f"Aggregation strategy: {aggregation_strategy}\n")
                f.write("\n")
            else:
                f.write(f"Best model type: Individual ({self.best_params.get('model_type', 'unknown')})\n")
                f.write(f"Compressor: {compressor_names[0]}\n")
                f.write(f"Compression metric: {compression_metric_names[0]}\n\n")
            
            # Parameter space
            f.write("PARAMETER SPACE:\n")
            f.write("-" * 30 + "\n")
            f.write("BASIC PARAMETERS:\n")
            f.write(f"  Compressor combinations ({len(self.parameter_space.compressor_names_combinations)}): {self.parameter_space.compressor_names_combinations}\n")
            f.write(f"  Metric combinations ({len(self.parameter_space.compression_metric_names_combinations)}): {self.parameter_space.compression_metric_names_combinations}\n")
            f.write(f"  Compression levels ({len(self.parameter_space.compression_levels)}): {self.parameter_space.compression_levels}\n")
            f.write(f"  Min size thresholds ({len(self.parameter_space.min_size_thresholds)}): {self.parameter_space.min_size_thresholds}\n")
            f.write(f"  Concat values ({len(self.parameter_space.concat_value)}): {self.parameter_space.concat_value}\n")
            f.write(f"  Use best sigma options: {self.parameter_space.use_best_sigma_options}\n")
            f.write(f"  Use symmetric matrix options: {self.parameter_space.use_symmetric_matrix_options}\n")
            f.write(f"  Model types: {self.parameter_space.model_types}\n")
            f.write(f"  Aggregation strategies ({len(self.parameter_space.agregation_strategy)}): {self.parameter_space.agregation_strategy}\n\n")
            
            f.write("ENSEMBLE PARAMETERS:\n")
            f.write(f"  Ensemble strategies ({len(self.parameter_space.ensemble_strategy)}): {self.parameter_space.ensemble_strategy}\n\n")
            
            f.write("MODEL-SPECIFIC PARAMETERS:\n")
            f.write(f"  ScOPE-OT:\n")
            f.write(f"    Matching metrics ({len(self.parameter_space.ot_matching_metrics)}): {self.parameter_space.ot_matching_metrics}\n")
            f.write(f"  ScOPE-PD:\n")
            f.write(f"    Distance metrics ({len(self.parameter_space.pd_distance_metrics)}): {self.parameter_space.pd_distance_metrics}\n\n")
            
            # Trial statistics
            completed_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            pruned_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])
            failed_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])
            
            f.write("TRIAL STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Completed trials: {completed_trials}\n")
            f.write(f"Pruned trials: {pruned_trials}\n")
            f.write(f"Failed trials: {failed_trials}\n")
            f.write(f"Total trials: {len(self.study.trials)}\n\n")
            
            # Ensemble vs Individual analysis
            df_results = self.get_trials_dataframe()
            if not df_results.empty:
                ensemble_trials = []
                individual_trials = []
                
                for idx, row in df_results.iterrows():
                    compressor_names = str(row.get('params_compressor_names', '')).split(',')
                    compression_metric_names = str(row.get('params_compression_metric_names', '')).split(',')
                    
                    is_ensemble = len(compressor_names) > 1 or len(compression_metric_names) > 1
                    
                    if is_ensemble:
                        ensemble_trials.append(row)
                    else:
                        individual_trials.append(row)
                
                f.write("ENSEMBLE VS INDIVIDUAL ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Ensemble trials: {len(ensemble_trials)}\n")
                f.write(f"Individual trials: {len(individual_trials)}\n")
                
                if ensemble_trials:
                    ensemble_df = pd.DataFrame(ensemble_trials)
                    f.write(f"Best ensemble score: {ensemble_df['value'].max():.6f}\n")
                
                if individual_trials:
                    individual_df = pd.DataFrame(individual_trials)
                    f.write(f"Best individual score: {individual_df['value'].max():.6f}\n")
                f.write("\n")
            
            # Best parameters
            f.write("BEST CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            for param, value in self.best_params.items():
                if param == 'concat_value':  # Updated from string_separator
                    value = repr(value)
                f.write(f"{param}: {value}\n")
            f.write("\n")
            
            # Parameter importance
            try:
                importances = optuna.importance.get_param_importances(self.study)
                f.write("PARAMETER IMPORTANCE:\n")
                f.write("-" * 30 + "\n")
                
                # Separate parameters by category
                basic_params = {}
                ot_params = {}
                pd_params = {}
                ensemble_params = {}
                
                for param, importance in importances.items():
                    if param.startswith('ot_'):
                        ot_params[param] = importance
                    elif param.startswith('pd_'):
                        pd_params[param] = importance
                    elif param in ['ensemble_strategy', 'agregation_strategy']:
                        ensemble_params[param] = importance
                    else:
                        basic_params[param] = importance
                
                # Write basic parameters
                if basic_params:
                    f.write("Basic Parameters:\n")
                    for param, importance in sorted(basic_params.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  {param}: {importance:.6f}\n")
                    f.write("\n")
                
                # Write ensemble parameters
                if ensemble_params:
                    f.write("Ensemble Parameters:\n")
                    for param, importance in sorted(ensemble_params.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  {param}: {importance:.6f}\n")
                    f.write("\n")
                
                # Write ScOPE-OT parameters
                if ot_params:
                    f.write("ScOPE-OT Parameters:\n")
                    for param, importance in sorted(ot_params.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  {param}: {importance:.6f}\n")
                    f.write("\n")
                
                # Write ScOPE-PD parameters
                if pd_params:
                    f.write("ScOPE-PD Parameters:\n")
                    for param, importance in sorted(pd_params.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  {param}: {importance:.6f}\n")
                    f.write("\n")
                
                # Write overall ranking
                f.write("Overall Ranking (All Parameters):\n")
                for param, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {param}: {importance:.6f}\n")
                f.write("\n")
                
            except Exception as e:
                f.write(f"PARAMETER IMPORTANCE: Could not calculate ({str(e)})\n\n")
            
            # Top 10 configurations
            if not df_results.empty:
                top_10 = df_results.nlargest(10, 'value')
                f.write("TOP 10 CONFIGURATIONS:\n")
                f.write("-" * 30 + "\n")
                
                for i, (idx, row) in enumerate(top_10.iterrows(), 1):
                    f.write(f"\nRank {i} (Trial {row['number']}):\n")
                    f.write(f"  Score: {row['value']:.6f}\n")
                    
                    # Extract parameters
                    param_cols = [col for col in df_results.columns if col.startswith('params_')]
                    for param_col in param_cols:
                        param_name = param_col.replace('params_', '')
                        param_value = row[param_col]
                        if param_name == 'concat_value':  # Updated from string_separator
                            param_value = repr(param_value)
                        elif pd.isna(param_value):
                            param_value = 'None'
                        f.write(f"  {param_name}: {param_value}\n")
        
        print(f"Analysis report saved to {filepath}")
    
    def save_top_results_csv(self, filename: Optional[str] = None, top_n: int = 10):
        """Save top N results to CSV file"""
        
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        if filename is None:
            filename = f"{self.study_name}_top{top_n}_{self.study_date}.csv"

        filepath = os.path.join(self.output_path, filename)
        
        df_results = self.get_trials_dataframe()
        if df_results.empty:
            print("No results to save.")
            return
        
        top_results = df_results.nlargest(top_n, 'value')
        
        cleaned_df = top_results.copy()
        column_mapping = {}
        for col in cleaned_df.columns:
            if col.startswith('params_'):
                new_name = col.replace('params_', '')
                column_mapping[col] = new_name
        
        cleaned_df = cleaned_df.rename(columns=column_mapping)
        
        important_cols = ['number', 'value', 'state']
        basic_param_cols = ['compressor_names', 'compression_metric_names', 'compression_level', 
                           'min_size_threshold', 'concat_value', 'use_best_sigma', 
                           'model_type', 'use_symmetric_matrix', 
                           'ensemble_strategy', 'agregation_strategy']
        
        model_specific_cols = []
        for col in cleaned_df.columns:
            if col.startswith('ot_') or col.startswith('pd_'):
                model_specific_cols.append(col)
        
        all_param_cols = basic_param_cols + model_specific_cols
        final_cols = important_cols + [col for col in all_param_cols if col in cleaned_df.columns]
        
        result_df = cleaned_df[final_cols]
        
        result_df.insert(0, 'rank', range(1, len(result_df) + 1))
        
        result_df.to_csv(filepath, index=False)
        print(f"Top {top_n} results saved to {filepath}")
        
        return result_df
    
    def save_complete_analysis(self, top_n: int = 10):
        """Save complete analysis: pickle, text report, and CSV"""
        
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        self.save_results()
        
        self.save_analysis_report()
        
        df_top = self.save_top_results_csv(top_n=top_n)
        
        print(f"\nComplete analysis saved for study: {self.study_name}")
        print(f"Output directory: {self.output_path}")
        print("Files created:")
        print(f"  - {self.study_name}_results_{self.study_date}.pkl (complete study data)")
        print(f"  - {self.study_name}_analysis_{self.study_date}.txt (detailed report)")
        print(f"  - {self.study_name}_top{top_n}_{self.study_date}.csv (top {top_n} configurations)")
        
        return df_top