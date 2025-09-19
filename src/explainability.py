import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. Install with: pip install lime")

class ModelExplainabilityEngine:
    """
    Comprehensive model explainability framework using SHAP, LIME, and custom interpretability methods.
    Provides both global and local explanations suitable for clinical decision support.
    """
    
    def __init__(self):
        self.explainers = {}
        self.explanations = {}
        self.feature_names = []
        self.model = None
        
    def comprehensive_explainability_analysis(self, model, X_train: pd.DataFrame, 
                                            X_test: pd.DataFrame, y_test: pd.Series = None,
                                            sample_size: int = 100) -> Dict:
        """
        Perform comprehensive model explainability analysis.
        """
        self.model = model
        self.feature_names = X_train.columns.tolist()
        
        explanations = {
            'global_explanations': {},
            'local_explanations': {},
            'feature_interactions': {},
            'model_behavior_analysis': {},
            'clinical_interpretations': {},
            'explanation_stability': {}
        }
        
        # Sample data for efficiency
        if len(X_test) > sample_size:
            sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_test_sample = X_test.iloc[sample_indices]
            y_test_sample = y_test.iloc[sample_indices] if y_test is not None else None
        else:
            X_test_sample = X_test
            y_test_sample = y_test
        
        # Global explanations
        if SHAP_AVAILABLE:
            global_shap = self._generate_global_shap_explanations(
                model, X_train, X_test_sample
            )
            explanations['global_explanations']['shap'] = global_shap
        
        # Permutation importance (model-agnostic)
        perm_importance = self._calculate_permutation_importance(
            model, X_test_sample, y_test_sample
        )
        explanations['global_explanations']['permutation_importance'] = perm_importance
        
        # Local explanations
        if SHAP_AVAILABLE:
            local_shap = self._generate_local_shap_explanations(
                model, X_train, X_test_sample.head(10)  # First 10 instances
            )
            explanations['local_explanations']['shap'] = local_shap
        
        if LIME_AVAILABLE:
            local_lime = self._generate_local_lime_explanations(
                model, X_train, X_test_sample.head(5)  # First 5 instances
            )
            explanations['local_explanations']['lime'] = local_lime
        
        # Feature interactions
        if SHAP_AVAILABLE:
            interactions = self._analyze_feature_interactions(
                model, X_train, X_test_sample
            )
            explanations['feature_interactions'] = interactions
        
        # Model behavior analysis
        behavior_analysis = self._analyze_model_behavior(
            model, X_test_sample, y_test_sample
        )
        explanations['model_behavior_analysis'] = behavior_analysis
        
        # Clinical interpretations
        clinical_interp = self._generate_clinical_interpretations(
            explanations, X_test_sample
        )
        explanations['clinical_interpretations'] = clinical_interp
        
        # Explanation stability
        stability = self._assess_explanation_stability(
            model, X_train, X_test_sample.head(20)
        )
        explanations['explanation_stability'] = stability
        
        self.explanations = explanations
        return explanations
    
    def _generate_global_shap_explanations(self, model, X_train: pd.DataFrame, 
                                         X_test: pd.DataFrame) -> Dict:
        """
        Generate global SHAP explanations.
        """
        try:
            # Choose appropriate explainer based on model type
            if hasattr(model, 'decision_function'):
                # Tree-based models
                if hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.LinearExplainer(model, X_train)
            else:
                # Use KernelExplainer as fallback (slower but model-agnostic)
                background_sample = shap.sample(X_train, min(100, len(X_train)))
                explainer = shap.KernelExplainer(model.predict_proba, background_sample)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary classification
            
            # Calculate global feature importance
            global_importance = np.abs(shap_values).mean(0)
            
            # Create feature importance ranking
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': global_importance
            }).sort_values('importance', ascending=False)
            
            self.explainers['shap'] = explainer
            
            return {
                'explainer_type': type(explainer).__name__,
                'shap_values': shap_values,
                'feature_importance': feature_importance_df.to_dict('records'),
                'global_importance_scores': global_importance,
                'summary_stats': {
                    'mean_abs_shap': np.mean(np.abs(shap_values)),
                    'std_abs_shap': np.std(np.abs(shap_values)),
                    'top_5_features': feature_importance_df.head(5)['feature'].tolist()
                }
            }
            
        except Exception as e:
            return {'error': f"SHAP global explanation failed: {str(e)}"}
    
    def _calculate_permutation_importance(self, model, X_test: pd.DataFrame, 
                                        y_test: pd.Series = None) -> Dict:
        """
        Calculate permutation-based feature importance.
        """
        try:
            from sklearn.inspection import permutation_importance
            
            if y_test is None:
                # Use model's own predictions as baseline
                y_test = model.predict(X_test)
            
            # Calculate permutation importance
            perm_result = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42
            )
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance_mean': perm_result.importances_mean,
                'importance_std': perm_result.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            return {
                'feature_importance': importance_df.to_dict('records'),
                'importance_scores': perm_result.importances_mean,
                'importance_std': perm_result.importances_std,
                'top_5_features': importance_df.head(5)['feature'].tolist()
            }
            
        except Exception as e:
            return {'error': f"Permutation importance calculation failed: {str(e)}"}
    
    def _generate_local_shap_explanations(self, model, X_train: pd.DataFrame,
                                        X_instances: pd.DataFrame) -> Dict:
        """
        Generate local SHAP explanations for specific instances.
        """
        try:
            if 'shap' not in self.explainers:
                return {'error': 'SHAP explainer not available'}
            
            explainer = self.explainers['shap']
            shap_values = explainer.shap_values(X_instances)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            
            # Organize explanations by instance
            local_explanations = []
            
            for i, (idx, instance) in enumerate(X_instances.iterrows()):
                # Get prediction
                prediction = model.predict_proba([instance])[0][1] if hasattr(model, 'predict_proba') else model.predict([instance])[0]
                
                # Get SHAP values for this instance
                instance_shap = shap_values[i]
                
                # Create feature contribution ranking
                contributions = pd.DataFrame({
                    'feature': self.feature_names,
                    'value': instance.values,
                    'shap_value': instance_shap,
                    'abs_shap_value': np.abs(instance_shap)
                }).sort_values('abs_shap_value', ascending=False)
                
                local_explanations.append({
                    'instance_index': int(idx),
                    'prediction': float(prediction),
                    'base_value': float(explainer.expected_value[1]) if hasattr(explainer, 'expected_value') and isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.0,
                    'feature_contributions': contributions.to_dict('records'),
                    'top_positive_contributors': contributions[contributions['shap_value'] > 0].head(3)['feature'].tolist(),
                    'top_negative_contributors': contributions[contributions['shap_value'] < 0].head(3)['feature'].tolist()
                })
            
            return {
                'local_explanations': local_explanations,
                'summary': {
                    'instances_explained': len(local_explanations),
                    'most_important_features': self._get_most_frequent_top_features(local_explanations)
                }
            }
            
        except Exception as e:
            return {'error': f"Local SHAP explanation failed: {str(e)}"}
    
    def _generate_local_lime_explanations(self, model, X_train: pd.DataFrame,
                                        X_instances: pd.DataFrame) -> Dict:
        """
        Generate local LIME explanations for specific instances.
        """
        try:
            # Initialize LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=self.feature_names,
                mode='classification',
                discretize_continuous=True
            )
            
            local_explanations = []
            
            for i, (idx, instance) in enumerate(X_instances.iterrows()):
                # Generate explanation
                explanation = explainer.explain_instance(
                    instance.values,
                    model.predict_proba,
                    num_features=len(self.feature_names),
                    num_samples=1000
                )
                
                # Extract explanation data
                lime_explanation = explanation.as_list()
                prediction = model.predict_proba([instance])[0][1]
                
                # Parse LIME explanation
                feature_contributions = []
                for feature_rule, weight in lime_explanation:
                    # Extract feature name from rule
                    feature_name = feature_rule.split()[0] if ' ' in feature_rule else feature_rule
                    feature_contributions.append({
                        'feature': feature_name,
                        'rule': feature_rule,
                        'weight': weight,
                        'abs_weight': abs(weight)
                    })
                
                # Sort by absolute weight
                feature_contributions.sort(key=lambda x: x['abs_weight'], reverse=True)
                
                local_explanations.append({
                    'instance_index': int(idx),
                    'prediction': float(prediction),
                    'feature_contributions': feature_contributions,
                    'lime_score': explanation.score,
                    'top_positive_contributors': [fc['feature'] for fc in feature_contributions if fc['weight'] > 0][:3],
                    'top_negative_contributors': [fc['feature'] for fc in feature_contributions if fc['weight'] < 0][:3]
                })
            
            return {
                'local_explanations': local_explanations,
                'summary': {
                    'instances_explained': len(local_explanations),
                    'avg_explanation_score': np.mean([exp['lime_score'] for exp in local_explanations])
                }
            }
            
        except Exception as e:
            return {'error': f"Local LIME explanation failed: {str(e)}"}
    
    def _analyze_feature_interactions(self, model, X_train: pd.DataFrame,
                                    X_test: pd.DataFrame) -> Dict:
        """
        Analyze feature interactions using SHAP interaction values.
        """
        try:
            if 'shap' not in self.explainers:
                return {'error': 'SHAP explainer not available'}
            
            explainer = self.explainers['shap']
            
            # Calculate interaction values (computationally expensive)
            sample_size = min(50, len(X_test))
            X_sample = X_test.head(sample_size)
            
            if hasattr(explainer, 'shap_interaction_values'):
                interaction_values = explainer.shap_interaction_values(X_sample)
            else:
                return {'message': 'Interaction analysis not supported for this explainer type'}
            
            # Handle multi-class output
            if len(interaction_values.shape) == 4:
                interaction_values = interaction_values[:, :, :, 1]  # Positive class
            
            # Calculate average interaction effects
            avg_interactions = np.abs(interaction_values).mean(0)
            
            # Find top interactions
            top_interactions = []
            n_features = len(self.feature_names)
            
            for i in range(n_features):
                for j in range(i+1, n_features):
                    interaction_strength = avg_interactions[i, j]
                    top_interactions.append({
                        'feature_1': self.feature_names[i],
                        'feature_2': self.feature_names[j],
                        'interaction_strength': float(interaction_strength)
                    })
            
            # Sort by interaction strength
            top_interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
            
            return {
                'interaction_matrix': avg_interactions,
                'top_10_interactions': top_interactions[:10],
                'strongest_interaction': top_interactions[0] if top_interactions else None,
                'analysis_sample_size': sample_size
            }
            
        except Exception as e:
            return {'error': f"Feature interaction analysis failed: {str(e)}"}
    
    def _analyze_model_behavior(self, model, X_test: pd.DataFrame, 
                              y_test: pd.Series = None) -> Dict:
        """
        Analyze overall model behavior patterns.
        """
        try:
            # Get predictions
            predictions = model.predict(X_test)
            prediction_probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            behavior_analysis = {
                'prediction_distribution': {
                    'positive_predictions': int(np.sum(predictions == 1)),
                    'negative_predictions': int(np.sum(predictions == 0)),
                    'positive_rate': float(np.mean(predictions))
                },
                'confidence_analysis': {},
                'decision_boundary_analysis': {},
                'prediction_consistency': {}
            }
            
            if prediction_probabilities is not None:
                # Confidence analysis
                confidence_stats = {
                    'mean_confidence': float(np.mean(prediction_probabilities)),
                    'std_confidence': float(np.std(prediction_probabilities)),
                    'high_confidence_predictions': int(np.sum((prediction_probabilities > 0.8) | (prediction_probabilities < 0.2))),
                    'uncertain_predictions': int(np.sum((prediction_probabilities >= 0.4) & (prediction_probabilities <= 0.6)))
                }
                behavior_analysis['confidence_analysis'] = confidence_stats
                
                # Decision boundary analysis
                boundary_analysis = {
                    'predictions_near_boundary': int(np.sum(np.abs(prediction_probabilities - 0.5) < 0.1)),
                    'clear_positive_predictions': int(np.sum(prediction_probabilities > 0.7)),
                    'clear_negative_predictions': int(np.sum(prediction_probabilities < 0.3))
                }
                behavior_analysis['decision_boundary_analysis'] = boundary_analysis
            
            # Feature value analysis for predictions
            feature_analysis = {}
            for feature in self.feature_names:
                if feature in X_test.columns:
                    positive_pred_values = X_test[predictions == 1][feature]
                    negative_pred_values = X_test[predictions == 0][feature]
                    
                    feature_analysis[feature] = {
                        'positive_pred_mean': float(positive_pred_values.mean()) if len(positive_pred_values) > 0 else None,
                        'negative_pred_mean': float(negative_pred_values.mean()) if len(negative_pred_values) > 0 else None,
                        'difference': float(positive_pred_values.mean() - negative_pred_values.mean()) if len(positive_pred_values) > 0 and len(negative_pred_values) > 0 else None
                    }
            
            behavior_analysis['feature_value_patterns'] = feature_analysis
            
            return behavior_analysis
            
        except Exception as e:
            return {'error': f"Model behavior analysis failed: {str(e)}"}
    
    def _generate_clinical_interpretations(self, explanations: Dict, 
                                         X_test: pd.DataFrame) -> Dict:
        """
        Generate clinical interpretations from explanations.
        """
        clinical_interp = {
            'risk_factor_hierarchy': [],
            'protective_factors': [],
            'clinical_decision_rules': [],
            'patient_archetypes': {},
            'actionable_insights': []
        }
        
        try:
            # Extract risk factor hierarchy
            if 'shap' in explanations.get('global_explanations', {}):
                shap_importance = explanations['global_explanations']['shap']['feature_importance']
                
                # Map features to clinical interpretations
                feature_mapping = {
                    'BMI': 'Body Mass Index',
                    'Age': 'Age',
                    'PhysActivity': 'Physical Activity',
                    'MentHlth': 'Mental Health Days',
                    'PhysHlth': 'Physical Health Days',
                    'Income': 'Socioeconomic Status',
                    'Sex': 'Gender',
                    'Fruits': 'Fruit Consumption',
                    'Veggies': 'Vegetable Consumption'
                }
                
                # Create risk factor hierarchy
                for feature_info in shap_importance[:10]:
                    feature = feature_info['feature']
                    importance = feature_info['importance']
                    
                    clinical_name = feature_mapping.get(feature, feature)
                    clinical_interp['risk_factor_hierarchy'].append({
                        'clinical_factor': clinical_name,
                        'technical_feature': feature,
                        'importance_score': importance,
                        'clinical_relevance': self._get_clinical_relevance(feature, importance)
                    })
            
            # Identify protective factors from local explanations
            if 'shap' in explanations.get('local_explanations', {}):
                local_shap = explanations['local_explanations']['shap']['local_explanations']
                
                protective_effects = {}
                for explanation in local_shap:
                    for contrib in explanation['feature_contributions']:
                        feature = contrib['feature']
                        shap_value = contrib['shap_value']
                        
                        if shap_value < 0:  # Negative SHAP = protective
                            if feature not in protective_effects:
                                protective_effects[feature] = []
                            protective_effects[feature].append(abs(shap_value))
                
                # Average protective effects
                for feature, effects in protective_effects.items():
                    if np.mean(effects) > 0.01:  # Threshold for meaningful effect
                        clinical_interp['protective_factors'].append({
                            'factor': feature_mapping.get(feature, feature),
                            'average_protective_effect': float(np.mean(effects)),
                            'consistency': float(len(effects) / len(local_shap))
                        })
            
            # Generate clinical decision rules
            clinical_interp['clinical_decision_rules'] = self._generate_decision_rules(
                explanations, X_test
            )
            
            # Generate actionable insights
            clinical_interp['actionable_insights'] = [
                "Focus interventions on highest-importance modifiable risk factors",
                "Consider patient-specific risk profiles for personalized treatment",
                "Monitor protective factors to maintain health benefits",
                "Implement multi-factor risk assessment protocols",
                "Develop targeted screening strategies for high-risk populations"
            ]
            
            return clinical_interp
            
        except Exception as e:
            return {'error': f"Clinical interpretation generation failed: {str(e)}"}
    
    def _get_clinical_relevance(self, feature: str, importance: float) -> str:
        """
        Get clinical relevance description for a feature.
        """
        relevance_map = {
            'BMI': 'Strong predictor - modifiable through diet and exercise',
            'Age': 'Non-modifiable risk factor - important for screening protocols',
            'PhysActivity': 'Highly modifiable - primary intervention target',
            'MentHlth': 'Modifiable through mental health interventions',
            'PhysHlth': 'May indicate underlying health conditions',
            'Income': 'Social determinant - affects access to care and resources',
            'Sex': 'Biological risk factor - influences screening recommendations',
            'Fruits': 'Modifiable dietary factor',
            'Veggies': 'Modifiable dietary factor'
        }
        
        base_relevance = relevance_map.get(feature, 'Clinical significance requires further evaluation')
        
        if importance > 0.1:
            return f"HIGH PRIORITY: {base_relevance}"
        elif importance > 0.05:
            return f"MODERATE PRIORITY: {base_relevance}"
        else:
            return f"LOW PRIORITY: {base_relevance}"
    
    def _generate_decision_rules(self, explanations: Dict, X_test: pd.DataFrame) -> List[str]:
        """
        Generate simple clinical decision rules from explanations.
        """
        rules = []
        
        try:
            # Extract top features from global explanations
            if 'shap' in explanations.get('global_explanations', {}):
                top_features = explanations['global_explanations']['shap']['summary_stats']['top_5_features']
                
                for feature in top_features[:3]:  # Top 3 features
                    if feature in X_test.columns:
                        feature_values = X_test[feature]
                        median_value = feature_values.median()
                        
                        if feature == 'BMI':
                            rules.append(f"If BMI > {median_value:.1f}, increase diabetes risk assessment priority")
                        elif feature == 'Age':
                            rules.append(f"Patients aged > {median_value:.0f} require enhanced screening")
                        elif feature == 'PhysActivity':
                            rules.append("Physical inactivity significantly increases diabetes risk")
                        else:
                            rules.append(f"Consider {feature} levels in risk stratification")
            
            return rules[:5]  # Limit to 5 rules
            
        except:
            return ["Clinical decision rules could not be automatically generated"]
    
    def _get_most_frequent_top_features(self, local_explanations: List[Dict]) -> List[str]:
        """
        Get the most frequently occurring top features across local explanations.
        """
        feature_counts = {}
        
        for explanation in local_explanations:
            top_features = explanation.get('top_positive_contributors', [])[:3]
            for feature in top_features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Sort by frequency
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        return [feature for feature, count in sorted_features[:5]]
    
    def _assess_explanation_stability(self, model, X_train: pd.DataFrame,
                                    X_instances: pd.DataFrame) -> Dict:
        """
        Assess stability of explanations across similar instances.
        """
        try:
            if not SHAP_AVAILABLE or 'shap' not in self.explainers:
                return {'error': 'SHAP not available for stability analysis'}
            
            explainer = self.explainers['shap']
            
            # Generate explanations multiple times with slight perturbations
            stability_results = []
            
            for idx, instance in X_instances.iterrows():
                instance_explanations = []
                
                # Generate explanations with slight noise
                for _ in range(5):  # 5 perturbations
                    noise = np.random.normal(0, 0.01, size=len(instance))
                    perturbed_instance = instance + noise
                    
                    try:
                        shap_values = explainer.shap_values(pd.DataFrame([perturbed_instance], columns=X_instances.columns))
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]
                        instance_explanations.append(shap_values[0])
                    except:
                        continue
                
                if len(instance_explanations) >= 3:
                    # Calculate stability as coefficient of variation
                    explanations_array = np.array(instance_explanations)
                    mean_explanation = np.mean(explanations_array, axis=0)
                    std_explanation = np.std(explanations_array, axis=0)
                    
                    # Stability score (lower is more stable)
                    stability_score = np.mean(std_explanation / (np.abs(mean_explanation) + 1e-8))
                    
                    stability_results.append({
                        'instance_index': int(idx),
                        'stability_score': float(stability_score),
                        'stable': stability_score < 0.5
                    })
            
            overall_stability = np.mean([r['stability_score'] for r in stability_results]) if stability_results else np.nan
            
            return {
                'instance_stability': stability_results,
                'overall_stability_score': float(overall_stability) if not np.isnan(overall_stability) else None,
                'stability_interpretation': 'Highly stable' if overall_stability < 0.3 else 'Moderately stable' if overall_stability < 0.7 else 'Unstable' if not np.isnan(overall_stability) else 'Could not assess',
                'stable_instances': len([r for r in stability_results if r['stable']])
            }
            
        except Exception as e:
            return {'error': f"Stability assessment failed: {str(e)}"}
    
    def create_explainability_dashboard(self, save_plots: bool = False) -> None:
        """
        Create comprehensive explainability dashboard.
        """
        if not self.explanations:
            print("No explanations available. Run comprehensive_explainability_analysis first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Explainability Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Global Feature Importance
        ax1 = axes[0, 0]
        if 'shap' in self.explanations['global_explanations']:
            importance_data = self.explanations['global_explanations']['shap']['feature_importance'][:10]
            features = [item['feature'] for item in importance_data]
            importances = [item['importance'] for item in importance_data]
            
            bars = ax1.barh(range(len(features)), importances, alpha=0.7, color='skyblue')
            ax1.set_yticks(range(len(features)))
            ax1.set_yticklabels(features)
            ax1.set_xlabel('SHAP Importance')
            ax1.set_title('Global Feature Importance (Top 10)')
            ax1.invert_yaxis()
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax1.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        # 2. Feature Interaction Heatmap
        ax2 = axes[0, 1]
        if 'interaction_matrix' in self.explanations['feature_interactions']:
            interaction_matrix = self.explanations['feature_interactions']['interaction_matrix']
            
            # Take subset for visualization
            n_features = min(8, len(self.feature_names))
            subset_matrix = interaction_matrix[:n_features, :n_features]
            subset_features = self.feature_names[:n_features]
            
            im = ax2.imshow(subset_matrix, cmap='RdBu_r', aspect='auto')
            ax2.set_xticks(range(n_features))
            ax2.set_yticks(range(n_features))
            ax2.set_xticklabels(subset_features, rotation=45, ha='right')
            ax2.set_yticklabels(subset_features)
            ax2.set_title('Feature Interactions Heatmap')
            
            # Add colorbar
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        # 3. Explanation Stability
        ax3 = axes[1, 0]
        if 'instance_stability' in self.explanations['explanation_stability']:
            stability_data = self.explanations['explanation_stability']['instance_stability']
            stability_scores = [item['stability_score'] for item in stability_data]
            
            ax3.hist(stability_scores, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.axvline(x=0.5, color='red', linestyle='--', label='Stability Threshold')
            ax3.set_xlabel('Stability Score (lower = more stable)')
            ax3.set_ylabel('Number of Instances')
            ax3.set_title('Explanation Stability Distribution')
            ax3.legend()
        
        # 4. Clinical Risk Factors
        ax4 = axes[1, 1]
        if 'risk_factor_hierarchy' in self.explanations['clinical_interpretations']:
            risk_factors = self.explanations['clinical_interpretations']['risk_factor_hierarchy'][:8]
            
            if risk_factors:
                clinical_names = [rf['clinical_factor'] for rf in risk_factors]
                importance_scores = [rf['importance_score'] for rf in risk_factors]
                
                colors = ['red' if score > 0.1 else 'orange' if score > 0.05 else 'green' 
                         for score in importance_scores]
                
                bars = ax4.barh(range(len(clinical_names)), importance_scores, 
                               alpha=0.7, color=colors)
                ax4.set_yticks(range(len(clinical_names)))
                ax4.set_yticklabels(clinical_names, fontsize=8)
                ax4.set_xlabel('Clinical Importance Score')
                ax4.set_title('Clinical Risk Factor Hierarchy')
                ax4.invert_yaxis()
                
                # Add priority labels
                for i, (bar, score) in enumerate(zip(bars, importance_scores)):
                    priority = 'HIGH' if score > 0.1 else 'MED' if score > 0.05 else 'LOW'
                    ax4.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                            priority, ha='left', va='center', fontsize=7, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('explainability_dashboard.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_explainability_report(self) -> str:
        """
        Generate comprehensive explainability report.
        """
        if not self.explanations:
            return "No explainability analysis results available."
        
        report = []
        report.append("MODEL EXPLAINABILITY ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Global Explanations
        report.append("\nGLOBAL MODEL EXPLANATIONS")
        report.append("-" * 35)
        
        if 'shap' in self.explanations['global_explanations']:
            shap_global = self.explanations['global_explanations']['shap']
            report.append(f"Explainer Type: {shap_global['explainer_type']}")
            
            report.append("\nTop 10 Most Important Features:")
            for i, feature_info in enumerate(shap_global['feature_importance'][:10], 1):
                report.append(f"{i:2d}. {feature_info['feature']}: {feature_info['importance']:.4f}")
        
        # Clinical Interpretations
        if 'clinical_interpretations' in self.explanations:
            clinical = self.explanations['clinical_interpretations']
            
            report.append("\nCLINICAL RISK FACTOR HIERARCHY")
            report.append("-" * 35)
            
            for rf in clinical.get('risk_factor_hierarchy', [])[:8]:
                report.append(f"{rf['clinical_factor']}")
                report.append(f"  Technical Feature: {rf['technical_feature']}")
                report.append(f"  Importance Score: {rf['importance_score']:.4f}")
                report.append(f"  Clinical Relevance: {rf['clinical_relevance']}")
                report.append("")
            
            if clinical.get('protective_factors'):
                report.append("PROTECTIVE FACTORS IDENTIFIED:")
                for pf in clinical['protective_factors'][:5]:
                    report.append(f"• {pf['factor']}: Avg protective effect = {pf['average_protective_effect']:.4f}")
                report.append("")
            
            if clinical.get('clinical_decision_rules'):
                report.append("CLINICAL DECISION RULES:")
                for i, rule in enumerate(clinical['clinical_decision_rules'], 1):
                    report.append(f"{i}. {rule}")
                report.append("")
        
        # Feature Interactions
        if 'top_10_interactions' in self.explanations.get('feature_interactions', {}):
            report.append("TOP FEATURE INTERACTIONS")
            report.append("-" * 25)
            
            interactions = self.explanations['feature_interactions']['top_10_interactions']
            for i, interaction in enumerate(interactions[:5], 1):
                report.append(f"{i}. {interaction['feature_1']} × {interaction['feature_2']}")
                report.append(f"   Interaction Strength: {interaction['interaction_strength']:.4f}")
        
        # Explanation Stability
        if 'overall_stability_score' in self.explanations.get('explanation_stability', {}):
            stability = self.explanations['explanation_stability']
            report.append("\nEXPLANATION STABILITY ASSESSMENT")
            report.append("-" * 35)
            report.append(f"Overall Stability Score: {stability['overall_stability_score']:.4f}")
            report.append(f"Stability Assessment: {stability['stability_interpretation']}")
            report.append(f"Stable Instances: {stability['stable_instances']}/{len(stability.get('instance_stability', []))}")
        
        # Model Behavior Analysis
        if 'prediction_distribution' in self.explanations.get('model_behavior_analysis', {}):
            behavior = self.explanations['model_behavior_analysis']
            report.append("\nMODEL BEHAVIOR ANALYSIS")
            report.append("-" * 25)
            
            pred_dist = behavior['prediction_distribution']
            report.append(f"Positive Prediction Rate: {pred_dist['positive_rate']:.2%}")
            report.append(f"Positive Predictions: {pred_dist['positive_predictions']:,}")
            report.append(f"Negative Predictions: {pred_dist['negative_predictions']:,}")
            
            if 'confidence_analysis' in behavior:
                conf = behavior['confidence_analysis']
                report.append(f"\nPrediction Confidence:")
                report.append(f"  Mean Confidence: {conf['mean_confidence']:.4f}")
                report.append(f"  High Confidence Predictions: {conf['high_confidence_predictions']:,}")
                report.append(f"  Uncertain Predictions: {conf['uncertain_predictions']:,}")
        
        # Summary and Recommendations
        report.append("\nSUMMARY AND RECOMMENDATIONS")
        report.append("-" * 35)
        
        if 'clinical_interpretations' in self.explanations:
            actionable_insights = self.explanations['clinical_interpretations'].get('actionable_insights', [])
            for i, insight in enumerate(actionable_insights, 1):
                report.append(f"{i}. {insight}")
        
        return "\n".join(report)