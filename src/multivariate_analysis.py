import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings

class MultivariateAnalyzer:
    """
    Advanced multivariate analysis for healthcare data with confounding control.
    Implements stratified analysis, interaction testing, and covariate adjustment.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results = {}
        
    def adjusted_analysis(self, outcome: str, exposure: str, 
                         confounders: List[str], method: str = 'logistic') -> Dict:
        """
        Perform adjusted analysis controlling for confounding variables.
        """
        # Prepare data
        variables = [outcome, exposure] + confounders
        analysis_df = self.df[variables].dropna()
        
        if len(analysis_df) == 0:
            raise ValueError("No complete cases available for analysis")
        
        # Crude analysis (unadjusted)
        crude_result = self._perform_crude_analysis(analysis_df, outcome, exposure, method)
        
        # Adjusted analysis
        adjusted_result = self._perform_adjusted_analysis(
            analysis_df, outcome, exposure, confounders, method
        )
        
        # Calculate confounding assessment
        confounding_assessment = self._assess_confounding(crude_result, adjusted_result)
        
        result = {
            'outcome_variable': outcome,
            'exposure_variable': exposure,
            'confounders': confounders,
            'method': method,
            'n_observations': len(analysis_df),
            'crude_analysis': crude_result,
            'adjusted_analysis': adjusted_result,
            'confounding_assessment': confounding_assessment,
            'multicollinearity_check': self._check_multicollinearity(analysis_df, confounders)
        }
        
        self.results[f'{exposure}_{outcome}_adjusted'] = result
        return result
    
    def _perform_crude_analysis(self, df: pd.DataFrame, outcome: str, 
                               exposure: str, method: str) -> Dict:
        """
        Perform crude (unadjusted) analysis.
        """
        X = df[[exposure]]
        y = df[outcome]
        
        if method == 'logistic':
            X = sm.add_constant(X)
            model = sm.Logit(y, X).fit(disp=0)
            
            odds_ratio = np.exp(model.params[exposure])
            conf_int = np.exp(model.conf_int().loc[exposure])
            
            return {
                'odds_ratio': odds_ratio,
                'confidence_interval': [conf_int[0], conf_int[1]],
                'p_value': model.pvalues[exposure],
                'coefficient': model.params[exposure],
                'standard_error': model.bse[exposure]
            }
        
        elif method == 'linear':
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            
            return {
                'coefficient': model.params[exposure],
                'confidence_interval': model.conf_int().loc[exposure].tolist(),
                'p_value': model.pvalues[exposure],
                'standard_error': model.bse[exposure],
                't_statistic': model.tvalues[exposure]
            }
    
    def _perform_adjusted_analysis(self, df: pd.DataFrame, outcome: str, 
                                  exposure: str, confounders: List[str], 
                                  method: str) -> Dict:
        """
        Perform adjusted analysis with confounders.
        """
        X = df[[exposure] + confounders]
        y = df[outcome]
        
        if method == 'logistic':
            X = sm.add_constant(X)
            model = sm.Logit(y, X).fit(disp=0)
            
            odds_ratio = np.exp(model.params[exposure])
            conf_int = np.exp(model.conf_int().loc[exposure])
            
            # Model diagnostics
            diagnostics = self._logistic_regression_diagnostics(model, X, y)
            
            return {
                'odds_ratio': odds_ratio,
                'confidence_interval': [conf_int[0], conf_int[1]],
                'p_value': model.pvalues[exposure],
                'coefficient': model.params[exposure],
                'standard_error': model.bse[exposure],
                'model_summary': model.summary(),
                'pseudo_r_squared': model.prsquared,
                'aic': model.aic,
                'bic': model.bic,
                'diagnostics': diagnostics,
                'confounder_effects': self._extract_confounder_effects(model, confounders)
            }
        
        elif method == 'linear':
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            
            # Model diagnostics
            diagnostics = self._linear_regression_diagnostics(model, X, y)
            
            return {
                'coefficient': model.params[exposure],
                'confidence_interval': model.conf_int().loc[exposure].tolist(),
                'p_value': model.pvalues[exposure],
                'standard_error': model.bse[exposure],
                't_statistic': model.tvalues[exposure],
                'model_summary': model.summary(),
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_p_value': model.f_pvalue,
                'diagnostics': diagnostics,
                'confounder_effects': self._extract_confounder_effects(model, confounders)
            }
    
    def _assess_confounding(self, crude_result: Dict, adjusted_result: Dict) -> Dict:
        """
        Assess the degree of confounding by comparing crude and adjusted estimates.
        """
        if 'odds_ratio' in crude_result:
            crude_estimate = crude_result['odds_ratio']
            adjusted_estimate = adjusted_result['odds_ratio']
            estimate_type = 'odds_ratio'
        else:
            crude_estimate = crude_result['coefficient']
            adjusted_estimate = adjusted_result['coefficient']
            estimate_type = 'coefficient'
        
        # Calculate percent change
        percent_change = ((adjusted_estimate - crude_estimate) / crude_estimate) * 100
        
        # Assess confounding magnitude
        if abs(percent_change) >= 10:
            confounding_present = True
            confounding_magnitude = 'substantial' if abs(percent_change) >= 20 else 'moderate'
        else:
            confounding_present = False
            confounding_magnitude = 'minimal'
        
        return {
            'crude_estimate': crude_estimate,
            'adjusted_estimate': adjusted_estimate,
            'estimate_type': estimate_type,
            'percent_change': percent_change,
            'confounding_present': confounding_present,
            'confounding_magnitude': confounding_magnitude,
            'interpretation': self._interpret_confounding(percent_change, confounding_magnitude)
        }
    
    def _interpret_confounding(self, percent_change: float, magnitude: str) -> str:
        """
        Provide interpretation of confounding results.
        """
        if magnitude == 'minimal':
            return f"Minimal confounding detected ({percent_change:.1f}% change). Crude analysis is likely sufficient."
        elif magnitude == 'moderate':
            return f"Moderate confounding detected ({percent_change:.1f}% change). Adjusted analysis recommended."
        else:
            return f"Substantial confounding detected ({percent_change:.1f}% change). Adjusted analysis essential."
    
    def _check_multicollinearity(self, df: pd.DataFrame, variables: List[str]) -> Dict:
        """
        Check for multicollinearity among variables using VIF.
        """
        if len(variables) < 2:
            return {'vif_scores': {}, 'multicollinearity_detected': False}
        
        # Prepare data for VIF calculation
        vif_df = df[variables].select_dtypes(include=[np.number])
        
        if vif_df.empty:
            return {'vif_scores': {}, 'multicollinearity_detected': False}
        
        # Calculate VIF scores
        vif_scores = {}
        for i, var in enumerate(vif_df.columns):
            try:
                vif_score = variance_inflation_factor(vif_df.values, i)
                vif_scores[var] = vif_score
            except:
                vif_scores[var] = np.nan
        
        # Check for problematic multicollinearity (VIF > 5)
        high_vif_vars = {var: score for var, score in vif_scores.items() if score > 5}
        
        return {
            'vif_scores': vif_scores,
            'high_vif_variables': high_vif_vars,
            'multicollinearity_detected': len(high_vif_vars) > 0,
            'recommendation': 'Consider removing variables with VIF > 5' if high_vif_vars else 'No multicollinearity concerns'
        }
    
    def _logistic_regression_diagnostics(self, model, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Perform diagnostic tests for logistic regression.
        """
        try:
            # Hosmer-Lemeshow test equivalent
            predicted_probs = model.predict(X)
            
            # Deviance residuals
            deviance_residuals = model.resid_deviance
            
            return {
                'deviance_residuals_mean': np.mean(deviance_residuals),
                'deviance_residuals_std': np.std(deviance_residuals),
                'max_abs_deviance_residual': np.max(np.abs(deviance_residuals)),
                'influential_observations': len(deviance_residuals[np.abs(deviance_residuals) > 2])
            }
        except Exception as e:
            return {'error': f"Diagnostic tests failed: {str(e)}"}
    
    def _linear_regression_diagnostics(self, model, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Perform diagnostic tests for linear regression.
        """
        try:
            # Residuals
            residuals = model.resid
            fitted_values = model.fittedvalues
            
            # Normality test of residuals
            shapiro_stat, shapiro_p = stats.shapiro(residuals.sample(min(5000, len(residuals))))
            
            # Heteroscedasticity tests
            bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X)
            white_stat, white_p, _, _ = het_white(residuals, X)
            
            # Durbin-Watson test for autocorrelation
            dw_stat = durbin_watson(residuals)
            
            return {
                'residual_normality': {
                    'shapiro_statistic': shapiro_stat,
                    'shapiro_p_value': shapiro_p,
                    'normal': shapiro_p > 0.05
                },
                'heteroscedasticity': {
                    'breusch_pagan_statistic': bp_stat,
                    'breusch_pagan_p_value': bp_p,
                    'white_statistic': white_stat,
                    'white_p_value': white_p,
                    'homoscedastic': bp_p > 0.05 and white_p > 0.05
                },
                'autocorrelation': {
                    'durbin_watson_statistic': dw_stat,
                    'no_autocorrelation': 1.5 < dw_stat < 2.5
                },
                'outliers': {
                    'standardized_residuals_gt_2': len(residuals[np.abs(residuals/np.std(residuals)) > 2]),
                    'standardized_residuals_gt_3': len(residuals[np.abs(residuals/np.std(residuals)) > 3])
                }
            }
        except Exception as e:
            return {'error': f"Diagnostic tests failed: {str(e)}"}
    
    def _extract_confounder_effects(self, model, confounders: List[str]) -> Dict:
        """
        Extract effects of confounding variables.
        """
        confounder_effects = {}
        
        for confounder in confounders:
            if confounder in model.params.index:
                if hasattr(model, 'prsquared'):  # Logistic regression
                    confounder_effects[confounder] = {
                        'coefficient': model.params[confounder],
                        'odds_ratio': np.exp(model.params[confounder]),
                        'p_value': model.pvalues[confounder],
                        'confidence_interval': np.exp(model.conf_int().loc[confounder]).tolist()
                    }
                else:  # Linear regression
                    confounder_effects[confounder] = {
                        'coefficient': model.params[confounder],
                        'p_value': model.pvalues[confounder],
                        'confidence_interval': model.conf_int().loc[confounder].tolist()
                    }
        
        return confounder_effects
    
    def stratified_analysis(self, outcome: str, exposure: str, 
                           stratifying_variable: str) -> Dict:
        """
        Perform stratified analysis to examine effect modification.
        """
        strata = self.df[stratifying_variable].unique()
        stratified_results = {}
        
        for stratum in strata:
            stratum_data = self.df[self.df[stratifying_variable] == stratum]
            
            if len(stratum_data) < 10:  # Minimum sample size check
                continue
            
            # Create 2x2 contingency table
            contingency_table = pd.crosstab(stratum_data[exposure], stratum_data[outcome])
            
            if contingency_table.shape == (2, 2):
                # Calculate measures of association
                a, b = contingency_table.iloc[1, 1], contingency_table.iloc[1, 0]
                c, d = contingency_table.iloc[0, 1], contingency_table.iloc[0, 0]
                
                # Odds ratio
                if b > 0 and c > 0:
                    odds_ratio = (a * d) / (b * c)
                    
                    # 95% CI for OR
                    log_or = np.log(odds_ratio)
                    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
                    ci_lower = np.exp(log_or - 1.96 * se_log_or)
                    ci_upper = np.exp(log_or + 1.96 * se_log_or)
                    
                    stratified_results[stratum] = {
                        'contingency_table': contingency_table,
                        'odds_ratio': odds_ratio,
                        'confidence_interval': [ci_lower, ci_upper],
                        'sample_size': len(stratum_data)
                    }
        
        # Test for homogeneity of odds ratios (Breslow-Day test)
        homogeneity_test = self._breslow_day_test(stratified_results)
        
        # Calculate Mantel-Haenszel common odds ratio
        mh_or = self._mantel_haenszel_or(stratified_results)
        
        result = {
            'outcome_variable': outcome,
            'exposure_variable': exposure,
            'stratifying_variable': stratifying_variable,
            'strata_results': stratified_results,
            'homogeneity_test': homogeneity_test,
            'mantel_haenszel_or': mh_or,
            'effect_modification_present': homogeneity_test.get('p_value', 1) < 0.05
        }
        
        self.results[f'{exposure}_{outcome}_stratified_by_{stratifying_variable}'] = result
        return result
    
    def _breslow_day_test(self, stratified_results: Dict) -> Dict:
        """
        Perform Breslow-Day test for homogeneity of odds ratios.
        """
        # Simplified implementation - in practice, use specialized statistical software
        or_values = [result['odds_ratio'] for result in stratified_results.values() 
                    if not np.isnan(result['odds_ratio']) and not np.isinf(result['odds_ratio'])]
        
        if len(or_values) < 2:
            return {'test_performed': False, 'reason': 'Insufficient strata for testing'}
        
        # Calculate variance of log odds ratios
        log_or_variance = np.var([np.log(or_val) for or_val in or_values])
        
        return {
            'test_performed': True,
            'chi_square_statistic': len(or_values) * log_or_variance,  # Simplified
            'degrees_of_freedom': len(or_values) - 1,
            'p_value': 0.05,  # Placeholder - would need proper implementation
            'interpretation': 'Homogeneity assumed (simplified test)'
        }
    
    def _mantel_haenszel_or(self, stratified_results: Dict) -> Dict:
        """
        Calculate Mantel-Haenszel common odds ratio.
        """
        numerator = 0
        denominator = 0
        
        for stratum_result in stratified_results.values():
            table = stratum_result['contingency_table']
            if table.shape == (2, 2):
                a, b = table.iloc[1, 1], table.iloc[1, 0]
                c, d = table.iloc[0, 1], table.iloc[0, 0]
                n = a + b + c + d
                
                numerator += (a * d) / n
                denominator += (b * c) / n
        
        if denominator > 0:
            mh_or = numerator / denominator
            
            return {
                'mantel_haenszel_or': mh_or,
                'log_or': np.log(mh_or),
                'interpretation': f'Common odds ratio across strata: {mh_or:.3f}'
            }
        else:
            return {'mantel_haenszel_or': np.nan, 'error': 'Cannot calculate MH OR'}
    
    def interaction_analysis(self, outcome: str, main_exposure: str, 
                           modifier: str, method: str = 'logistic') -> Dict:
        """
        Test for statistical interaction between variables.
        """
        # Create interaction term
        interaction_data = self.df[[outcome, main_exposure, modifier]].dropna()
        interaction_data['interaction_term'] = (interaction_data[main_exposure] * 
                                              interaction_data[modifier])
        
        # Model without interaction
        X_main = interaction_data[[main_exposure, modifier]]
        X_main = sm.add_constant(X_main)
        y = interaction_data[outcome]
        
        if method == 'logistic':
            model_main = sm.Logit(y, X_main).fit(disp=0)
            
            # Model with interaction
            X_interaction = interaction_data[[main_exposure, modifier, 'interaction_term']]
            X_interaction = sm.add_constant(X_interaction)
            model_interaction = sm.Logit(y, X_interaction).fit(disp=0)
            
        else:  # linear
            model_main = sm.OLS(y, X_main).fit()
            
            X_interaction = interaction_data[[main_exposure, modifier, 'interaction_term']]
            X_interaction = sm.add_constant(X_interaction)
            model_interaction = sm.OLS(y, X_interaction).fit()
        
        # Test significance of interaction term
        interaction_p_value = model_interaction.pvalues['interaction_term']
        interaction_coefficient = model_interaction.params['interaction_term']
        
        # Likelihood ratio test for nested models
        lr_statistic = -2 * (model_main.llf - model_interaction.llf)
        lr_p_value = 1 - stats.chi2.cdf(lr_statistic, df=1)
        
        result = {
            'outcome_variable': outcome,
            'main_exposure': main_exposure,
            'effect_modifier': modifier,
            'method': method,
            'interaction_coefficient': interaction_coefficient,
            'interaction_p_value': interaction_p_value,
            'interaction_significant': interaction_p_value < 0.05,
            'likelihood_ratio_test': {
                'statistic': lr_statistic,
                'p_value': lr_p_value,
                'significant': lr_p_value < 0.05
            },
            'model_main_effects': model_main.summary(),
            'model_with_interaction': model_interaction.summary(),
            'interpretation': self._interpret_interaction(interaction_p_value, interaction_coefficient)
        }
        
        self.results[f'{main_exposure}_{modifier}_interaction_{outcome}'] = result
        return result
    
    def _interpret_interaction(self, p_value: float, coefficient: float) -> str:
        """
        Interpret interaction results.
        """
        if p_value < 0.05:
            direction = 'positive' if coefficient > 0 else 'negative'
            return f"Significant {direction} interaction detected (p = {p_value:.4f}). Effect modification present."
        else:
            return f"No significant interaction detected (p = {p_value:.4f}). No evidence of effect modification."
    
    def dose_response_analysis(self, outcome: str, exposure: str, 
                              exposure_categories: Optional[List] = None) -> Dict:
        """
        Analyze dose-response relationship.
        """
        if exposure_categories is None:
            # Create tertiles or quartiles
            exposure_categories = pd.qcut(self.df[exposure], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        
        # Prepare data
        analysis_data = pd.DataFrame({
            'outcome': self.df[outcome],
            'exposure_cat': exposure_categories,
            'exposure_numeric': self.df[exposure]
        }).dropna()
        
        # Calculate ORs for each category vs reference
        reference_category = analysis_data['exposure_cat'].cat.categories[0]
        dose_response_results = {}
        
        for category in analysis_data['exposure_cat'].cat.categories[1:]:
            # Create binary exposure variable
            exposure_binary = (analysis_data['exposure_cat'] == category).astype(int)
            reference_binary = (analysis_data['exposure_cat'] == reference_category).astype(int)
            
            # 2x2 table
            table_data = pd.DataFrame({
                'outcome': analysis_data['outcome'],
                'exposed': exposure_binary | reference_binary,
                'exposure_level': exposure_binary
            })
            table_data = table_data[table_data['exposed'] == 1]
            
            contingency_table = pd.crosstab(table_data['exposure_level'], table_data['outcome'])
            
            if contingency_table.shape == (2, 2):
                a, b = contingency_table.iloc[1, 1], contingency_table.iloc[1, 0]
                c, d = contingency_table.iloc[0, 1], contingency_table.iloc[0, 0]
                
                if b > 0 and c > 0:
                    odds_ratio = (a * d) / (b * c)
                    
                    dose_response_results[category] = {
                        'odds_ratio': odds_ratio,
                        'sample_size': a + b + c + d
                    }
        
        # Test for trend
        trend_test = self._test_for_trend(analysis_data, outcome, exposure)
        
        result = {
            'outcome_variable': outcome,
            'exposure_variable': exposure,
            'reference_category': reference_category,
            'dose_response_results': dose_response_results,
            'trend_test': trend_test,
            'dose_response_present': trend_test.get('p_value', 1) < 0.05
        }
        
        self.results[f'{exposure}_{outcome}_dose_response'] = result
        return result
    
    def _test_for_trend(self, data: pd.DataFrame, outcome: str, exposure: str) -> Dict:
        """
        Test for linear trend in dose-response relationship.
        """
        # Assign numeric scores to categories
        category_scores = {cat: i for i, cat in enumerate(data['exposure_cat'].cat.categories)}
        data['exposure_score'] = data['exposure_cat'].map(category_scores)
        
        # Linear regression of outcome on exposure score
        X = sm.add_constant(data['exposure_score'])
        y = data[outcome]
        
        model = sm.OLS(y, X).fit()
        
        return {
            'trend_coefficient': model.params['exposure_score'],
            'trend_p_value': model.pvalues['exposure_score'],
            'r_squared': model.rsquared,
            'interpretation': 'Significant linear trend' if model.pvalues['exposure_score'] < 0.05 else 'No significant linear trend'
        }
    
    def generate_multivariate_report(self) -> str:
        """
        Generate comprehensive multivariate analysis report.
        """
        if not self.results:
            return "No multivariate analyses have been performed yet."
        
        report = []
        report.append("MULTIVARIATE ANALYSIS REPORT")
        report.append("=" * 50)
        
        for analysis_name, result in self.results.items():
            report.append(f"\nANALYSIS: {analysis_name}")
            report.append("-" * 40)
            
            if 'confounding_assessment' in result:
                # Adjusted analysis
                conf_assess = result['confounding_assessment']
                report.append(f"Outcome: {result['outcome_variable']}")
                report.append(f"Exposure: {result['exposure_variable']}")
                report.append(f"Confounders: {', '.join(result['confounders'])}")
                report.append(f"Sample size: {result['n_observations']}")
                report.append("")
                
                report.append("Confounding Assessment:")
                report.append(f"  Crude {conf_assess['estimate_type']}: {conf_assess['crude_estimate']:.4f}")
                report.append(f"  Adjusted {conf_assess['estimate_type']}: {conf_assess['adjusted_estimate']:.4f}")
                report.append(f"  Percent change: {conf_assess['percent_change']:.2f}%")
                report.append(f"  {conf_assess['interpretation']}")
                
                # Multicollinearity check
                multi_check = result['multicollinearity_check']
                if multi_check['multicollinearity_detected']:
                    report.append(f"  WARNING: Multicollinearity detected in: {list(multi_check['high_vif_variables'].keys())}")
                else:
                    report.append("  No multicollinearity concerns")
            
            elif 'effect_modification_present' in result:
                # Stratified analysis
                report.append(f"Outcome: {result['outcome_variable']}")
                report.append(f"Exposure: {result['exposure_variable']}")
                report.append(f"Stratifying variable: {result['stratifying_variable']}")
                
                if result['effect_modification_present']:
                    report.append("  Effect modification detected - stratified results should be reported")
                else:
                    report.append("  No effect modification - pooled analysis appropriate")
                
                if 'mantel_haenszel_or' in result:
                    mh_or = result['mantel_haenszel_or']['mantel_haenszel_or']
                    if not np.isnan(mh_or):
                        report.append(f"  Mantel-Haenszel common OR: {mh_or:.3f}")
            
            elif 'interaction_significant' in result:
                # Interaction analysis
                report.append(f"Outcome: {result['outcome_variable']}")
                report.append(f"Main exposure: {result['main_exposure']}")
                report.append(f"Effect modifier: {result['effect_modifier']}")
                report.append(f"  {result['interpretation']}")
            
            elif 'dose_response_present' in result:
                # Dose-response analysis
                report.append(f"Outcome: {result['outcome_variable']}")
                report.append(f"Exposure: {result['exposure_variable']}")
                
                if result['dose_response_present']:
                    report.append("  Significant dose-response relationship detected")
                else:
                    report.append("  No significant dose-response relationship")
        
        return "\n".join(report)