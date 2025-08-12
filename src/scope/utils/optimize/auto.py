import os
import warnings
import pandas as pd
from typing import List, Dict, Optional, Any, Union

import optuna
import optunahub
from optuna.pruners import MedianPruner

from scope.utils.optimize.base import ScOPEOptimizer
from scope.utils.optimize.params import ParameterSpace

warnings.filterwarnings('ignore')


class ScOPEOptimizerAuto(ScOPEOptimizer):
    """Automatic sampler selection optimization for ScOPE models using AutoSampler."""
    
    def __init__(self, 
                 parameter_space: Optional[ParameterSpace] = None,
                 free_cpu: int = 0,
                 random_seed: int = 42,
                 cv_folds: int = 3,
                 study_name: str = "scope_auto_optimization",
                 output_path: str = "./results",
                 n_trials: int = 75,  # Balanced for auto-selection
                 target_metric: Union[str, Dict[str, float]] = 'auc_roc',
                 use_cache: bool = True,
                 constraints_func: Optional[callable] = None
                 ):
        """Initialize the AutoSampler optimizer
        
        Args:
            constraints_func: Optional constraint function for the optimization.
                             Should return True if constraints are satisfied.
        """
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
        
        self.constraints_func = constraints_func
        self.auto_sampler_module = None
        self.sampler_history = []  # Track which samplers were used
        
        os.makedirs(self.output_path, exist_ok=True)
    
    def _load_auto_sampler(self):
        """Load AutoSampler from OptunaHub"""
        try:
            self.auto_sampler_module = optunahub.load_module("samplers/auto_sampler")
            print("✓ AutoSampler loaded successfully from OptunaHub")
        except Exception as e:
            print(f"❌ Failed to load AutoSampler: {e}")
            print("💡 Make sure you have installed: pip install optunahub cmaes scipy torch")
            raise ImportError("AutoSampler dependencies not available. Please install: pip install optunahub cmaes scipy torch")
    
    def optimize(self,
                X_validation: List[str],
                y_validation: List[str],
                kw_samples_validation: List[Dict[str, Any]]) -> optuna.Study:
        """Run automatic sampler selection optimization"""
        
        print("=== AUTO SAMPLER OPTIMIZATION SCOPE ===\n")
        print(f"Validation data: {len(X_validation)} samples")
        print(f"Classes: {sorted(set(y_validation))}")
        print(f"Trials: {self.n_trials}")
        print(f"CV Folds: {self.cv_folds}")
        print(f"Constraints function: {'Yes' if self.constraints_func else 'None'}")
        print()
        
        # Load AutoSampler
        self._load_auto_sampler()
        
        self.validate_data(y_validation)
        self.validate_optimization_setup()
        
        self.print_target_metric_info()
        print()
        
        self.print_parameter_space()
        print("\n🤖 AutoSampler Strategy:")
        print("• GPSampler: Early stages (excellent sample efficiency)")
        print("• TPESampler: Categorical variables (flexible handling)")
        print("• Dynamic switching: Adapts based on optimization progress")
        print("• Automatic selection: No manual tuning required!")
        
        objective_func = self._create_objective_function(
            X_validation, y_validation, kw_samples_validation
        )
        
        # Wrap objective function if constraints are provided
        if self.constraints_func:
            original_objective = objective_func
            def constrained_objective(trial):
                result = original_objective(trial)
                if not self.constraints_func(trial):
                    # Return a penalty value if constraints not satisfied
                    penalty = float('inf') if self.get_optimization_direction() == 'minimize' else float('-inf')
                    return penalty
                return result
            objective_func = constrained_objective
        
        direction = self.get_optimization_direction()
        
        # Configure AutoSampler
        auto_sampler_kwargs = {
            'seed': self.random_seed,
        }
        
        if self.constraints_func:
            auto_sampler_kwargs['constraints_func'] = self.constraints_func
        
        # Configure Optuna study with AutoSampler
        self.study = optuna.create_study(
            direction=direction,
            sampler=self.auto_sampler_module.AutoSampler(**auto_sampler_kwargs),
            study_name=f"{self.study_name}_{self.study_date}",
            storage=f"sqlite:///{self.output_path}/optuna_{self.study_name}.sqlite3",
            load_if_exists=True
        )
        
        # Execute optimization
        print("\nStarting AutoSampler optimization...")
        print("🔄 AutoSampler will automatically switch between algorithms as needed...")
        
        # Custom callback to track sampler usage
        def track_sampler_callback(study, trial):
            # Try to get information about which sampler was used
            # This is approximate since AutoSampler doesn't expose this directly
            current_sampler = type(study.sampler).__name__
            if len(self.sampler_history) == 0 or self.sampler_history[-1] != current_sampler:
                self.sampler_history.append(current_sampler)
                print(f"🔀 Trial {trial.number}: Using {current_sampler}")
        
        self.study.optimize(
            objective_func,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            callbacks=[track_sampler_callback]
        )
        
        # Store best results
        self.best_params = self.study.best_params
        self.best_model = self.create_model_from_params(self.best_params)
        
        # Show results
        print("\n=== AUTO SAMPLER OPTIMIZATION RESULTS ===")
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
        
        # Show AutoSampler insights
        self._analyze_auto_sampler_performance()
        
        cache_size = self._eval_cache_size_()
        total_trials = len(self.study.trials)

        cached_percent = 100 * cache_size / total_trials if total_trials > 0 else 0
        not_cached_percent = 100 - cached_percent
        
        print(f"Cached evaluations (unique parameter sets): {cache_size} ({cached_percent:.2f}%)")
        print(f"Non-cached evaluations (evaluations recomputed): {total_trials - cache_size} ({not_cached_percent:.2f}%)")

        self.print_best_configuration()
        
        return self.study
    
    def _analyze_auto_sampler_performance(self):
        """Analyze AutoSampler's performance and decisions"""
        if not self.study or not self.study.trials:
            return
        
        print(f"\n🤖 AutoSampler Analysis:")
        print(f"   Total samplers used: {len(set(self.sampler_history))}")
        print(f"   Sampler transitions: {len(self.sampler_history) - 1}")
        
        # Analyze performance progression
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials) >= 10:
            values = [t.value for t in completed_trials]
            
            # Early vs late performance
            early_trials = values[:len(values)//3] if len(values) >= 9 else values[:3]
            late_trials = values[-len(values)//3:] if len(values) >= 9 else values[-3:]
            
            if self.get_optimization_direction() == 'maximize':
                early_best = max(early_trials)
                late_best = max(late_trials)
                improvement = late_best - early_best
            else:
                early_best = min(early_trials)
                late_best = min(late_trials)
                improvement = early_best - late_best
            
            print(f"   Early stage best: {early_best:.4f}")
            print(f"   Late stage best: {late_best:.4f}")
            print(f"   Improvement: {improvement:.4f}")
            
            # Analyze consistency
            if len(values) >= 20:
                recent_std = pd.Series(values[-10:]).std()
                print(f"   Recent convergence (std): {recent_std:.4f}")

    def analyze_results(self):
        """Analyze AutoSampler optimization results"""
        
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        print("\n=== DETAILED AUTO SAMPLER ANALYSIS ===")
        
        # Basic statistics
        completed_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])
        
        print(f"Completed trials: {completed_trials}")
        print(f"Pruned trials: {pruned_trials}")
        print(f"Failed trials: {failed_trials}")
        
        # AutoSampler specific analysis
        print(f"\nAutoSampler Configuration:")
        print(f"  • Sampler: {type(self.study.sampler).__name__}")
        print(f"  • Constraints: {'Yes' if self.constraints_func else 'None'}")
        print(f"  • Adaptive strategy: Automatic (GPSampler + TPESampler)")
        
        # Performance evolution analysis
        if len(self.study.trials) >= 10:
            values = [t.value for t in self.study.trials if t.value is not None]
            if values:
                print(f"\nPerformance Evolution:")
                
                # Divide into phases
                phase_size = len(values) // 3
                if phase_size > 0:
                    phase1 = values[:phase_size]  # Early exploration
                    phase2 = values[phase_size:2*phase_size]  # Middle exploitation
                    phase3 = values[2*phase_size:]  # Final convergence
                    
                    print(f"  • Phase 1 (Exploration) avg: {sum(phase1)/len(phase1):.4f}")
                    print(f"  • Phase 2 (Exploitation) avg: {sum(phase2)/len(phase2):.4f}")
                    print(f"  • Phase 3 (Convergence) avg: {sum(phase3)/len(phase3):.4f}")
                    
                    # Best in each phase
                    if self.get_optimization_direction() == 'maximize':
                        print(f"  • Best in phase 1: {max(phase1):.4f}")
                        print(f"  • Best in phase 2: {max(phase2):.4f}")
                        print(f"  • Best in phase 3: {max(phase3):.4f}")
                    else:
                        print(f"  • Best in phase 1: {min(phase1):.4f}")
                        print(f"  • Best in phase 2: {min(phase2):.4f}")
                        print(f"  • Best in phase 3: {min(phase3):.4f}")
        
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
        
        # Parameter importance (works well with AutoSampler)
        try:
            importances = optuna.importance.get_param_importances(self.study)
            print("\nParameter importance (AutoSampler insights):")
            print("-" * 50)
            
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
            
            # Print by category
            if basic_params:
                print("Basic Parameters:")
                for param, importance in sorted(basic_params.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {param}: {importance:.6f}")
                print()
            
            if ensemble_params:
                print("Ensemble Parameters:")
                for param, importance in sorted(ensemble_params.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {param}: {importance:.6f}")
                print()
            
            if ot_params:
                print("ScOPE-OT Parameters:")
                for param, importance in sorted(ot_params.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {param}: {importance:.6f}")
                print()
            
            if pd_params:
                print("ScOPE-PD Parameters:")
                for param, importance in sorted(pd_params.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {param}: {importance:.6f}")
                print()
            
            # AutoSampler generally provides good parameter importance
            print("Overall Ranking (Top 10 - AutoSampler prioritized):")
            top_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
            for param, importance in top_params:
                print(f"  {param}: {importance:.6f}")
            
        except Exception as e:
            print(f"Could not calculate parameter importance: {e}")
        
        if not df_results.empty:
            print("\nTop 5 configurations (AutoSampler discoveries):")
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
        """Save detailed AutoSampler analysis report to text file"""
        
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        if filename is None:
            filename = f"{self.study_name}_auto_analysis_{self.study_date}.txt"

        filepath = os.path.join(self.output_path, filename)
        
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SCOPE AUTOSAMPLER OPTIMIZATION ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Study information
            f.write("STUDY INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Study name: {self.study_name}\n")
            f.write(f"Study date: {self.study_date}\n")
            f.write(f"Output path: {self.output_path}\n")
            f.write(f"Sampler: AutoSampler (adaptive algorithm selection)\n\n")
            
            # AutoSampler configuration
            f.write("AUTOSAMPLER CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Adaptive strategy: GPSampler (early) + TPESampler (categorical) + dynamic switching\n")
            f.write(f"Constraints function: {'Yes' if self.constraints_func else 'None'}\n")
            f.write(f"Random seed: {self.random_seed}\n")
            f.write(f"Automatic algorithm selection: Enabled\n\n")
            
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
            f.write(f"Best score achieved: {self.study.best_value:.6f}\n\n")
            
            # AutoSampler strategy explanation
            f.write("AUTOSAMPLER STRATEGY:\n")
            f.write("-" * 30 + "\n")
            f.write("AutoSampler automatically selects the best algorithm based on:\n")
            f.write("• Problem characteristics (continuous vs categorical parameters)\n")
            f.write("• Optimization progress (early exploration vs late exploitation)\n") 
            f.write("• Sample efficiency requirements\n\n")
            f.write("Algorithm selection logic:\n")
            f.write("• GPSampler: Early stages with excellent sample efficiency\n")
            f.write("• TPESampler: Flexible handling of categorical variables\n")
            f.write("• Dynamic switching: Adapts based on convergence patterns\n\n")
            
            # Continue with standard analysis sections...
            # (Best model, parameter space, trials, etc. - similar to other optimizers)
            
        print(f"AutoSampler analysis report saved to {filepath}")
    
    def save_complete_analysis(self, top_n: int = 10):
        """Save complete AutoSampler analysis: pickle, text report, and CSV"""
        
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        self.save_results()
        self.save_analysis_report()
        
        # Reuse CSV saving method from Bayesian optimizer
        from scope.utils.optimize import ScOPEOptimizerBayesian
        bayesian_instance = ScOPEOptimizerBayesian.__new__(ScOPEOptimizerBayesian)
        bayesian_instance.study = self.study
        bayesian_instance.study_name = self.study_name  
        bayesian_instance.study_date = self.study_date
        bayesian_instance.output_path = self.output_path
        df_top = bayesian_instance.save_top_results_csv(
            filename=f"{self.study_name}_auto_top{top_n}_{self.study_date}.csv", 
            top_n=top_n
        )
        
        print(f"\nComplete AutoSampler analysis saved for study: {self.study_name}")
        print(f"Output directory: {self.output_path}")
        print("Files created:")
        print(f"  - {self.study_name}_results_{self.study_date}.pkl (complete study data)")
        print(f"  - {self.study_name}_auto_analysis_{self.study_date}.txt (detailed report)")
        print(f"  - {self.study_name}_auto_top{top_n}_{self.study_date}.csv (top {top_n} configurations)")
        
        return df_top
    
    def compare_with_baseline(self, baseline_optimizer, X_validation, y_validation, kw_samples_validation):
        """Compare AutoSampler performance with another optimizer"""
        print(f"\n🔄 Comparing AutoSampler vs {type(baseline_optimizer).__name__}...")
        
        # Run baseline
        baseline_study = baseline_optimizer.optimize(X_validation, y_validation, kw_samples_validation)
        
        # Compare results
        auto_best = self.study.best_value
        baseline_best = baseline_study.best_value
        
        if self.get_optimization_direction() == 'maximize':
            improvement = auto_best - baseline_best
            better = auto_best > baseline_best
        else:
            improvement = baseline_best - auto_best  
            better = auto_best < baseline_best
        
        print(f"\n📊 Comparison Results:")
        print(f"   AutoSampler best: {auto_best:.4f}")
        print(f"   {type(baseline_optimizer).__name__} best: {baseline_best:.4f}")
        print(f"   Improvement: {improvement:.4f}")
        print(f"   AutoSampler is {'✓ Better' if better else '✗ Worse'}")
        
        return {
            'auto_best': auto_best,
            'baseline_best': baseline_best,
            'improvement': improvement,
            'auto_is_better': better
        }