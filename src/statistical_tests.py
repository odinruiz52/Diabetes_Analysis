import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from typing import Dict, Tuple, List
import warnings

class StatisticalAnalyzer:
    """
    Comprehensive statistical testing framework for healthcare data analysis.
    Provides statistical significance testing with proper p-value corrections.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results = {}
    
    def chi_square_test(self, df: pd.DataFrame, var1: str, var2: str) -> Dict:
        """
        Perform chi-square test of independence for categorical variables.
        """
        contingency_table = pd.crosstab(df[var1], df[var2])
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate effect size (Cramer's V)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
        
        result = {
            'test_type': 'Chi-Square Test',
            'variables': [var1, var2],
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'effect_size_cramers_v': cramers_v,
            'significant': p_value < self.alpha,
            'contingency_table': contingency_table,
            'expected_frequencies': expected
        }
        
        self.results[f'{var1}_vs_{var2}_chi2'] = result
        return result
    
    def mann_whitney_u_test(self, df: pd.DataFrame, group_var: str, value_var: str) -> Dict:
        """
        Perform Mann-Whitney U test for non-parametric comparison of two groups.
        """
        groups = df[group_var].unique()
        if len(groups) != 2:
            raise ValueError("Mann-Whitney U test requires exactly 2 groups")
        
        group1_data = df[df[group_var] == groups[0]][value_var]
        group2_data = df[df[group_var] == groups[1]][value_var]
        
        statistic, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        
        # Calculate effect size (r = Z / sqrt(N))
        z_score = stats.norm.ppf(1 - p_value/2)
        n = len(group1_data) + len(group2_data)
        effect_size = abs(z_score) / np.sqrt(n)
        
        result = {
            'test_type': 'Mann-Whitney U Test',
            'group_variable': group_var,
            'value_variable': value_var,
            'groups': list(groups),
            'u_statistic': statistic,
            'p_value': p_value,
            'effect_size_r': effect_size,
            'significant': p_value < self.alpha,
            'group1_median': group1_data.median(),
            'group2_median': group2_data.median(),
            'group1_n': len(group1_data),
            'group2_n': len(group2_data)
        }
        
        self.results[f'{group_var}_{value_var}_mannwhitney'] = result
        return result
    
    def kruskal_wallis_test(self, df: pd.DataFrame, group_var: str, value_var: str) -> Dict:
        """
        Perform Kruskal-Wallis test for non-parametric comparison of multiple groups.
        """
        groups = df[group_var].unique()
        group_data = [df[df[group_var] == group][value_var] for group in groups]
        
        statistic, p_value = kruskal(*group_data)
        
        # Calculate effect size (eta-squared)
        n = len(df)
        eta_squared = (statistic - len(groups) + 1) / (n - len(groups))
        
        result = {
            'test_type': 'Kruskal-Wallis Test',
            'group_variable': group_var,
            'value_variable': value_var,
            'groups': list(groups),
            'h_statistic': statistic,
            'p_value': p_value,
            'effect_size_eta_squared': eta_squared,
            'significant': p_value < self.alpha,
            'group_medians': {group: df[df[group_var] == group][value_var].median() 
                            for group in groups}
        }
        
        # Post-hoc analysis if significant
        if p_value < self.alpha and len(groups) > 2:
            try:
                posthoc = pairwise_tukeyhsd(df[value_var], df[group_var])
                result['posthoc_analysis'] = posthoc.summary()
            except Exception:
                result['posthoc_analysis'] = "Post-hoc analysis failed"
        
        self.results[f'{group_var}_{value_var}_kruskal'] = result
        return result
    
    def logistic_regression_analysis(self, df: pd.DataFrame, outcome_var: str, 
                                   predictors: List[str]) -> Dict:
        """
        Perform logistic regression analysis with odds ratios and confidence intervals.
        """
        # Prepare data
        model_data = df[[outcome_var] + predictors].dropna()
        
        # Convert categorical predictors to dummy variables
        X = pd.get_dummies(model_data[predictors], drop_first=True)
        y = model_data[outcome_var]
        
        # Add constant
        X = sm.add_constant(X)
        
        # Fit model
        model = sm.Logit(y, X).fit(disp=0)
        
        # Calculate odds ratios and confidence intervals
        odds_ratios = np.exp(model.params)
        conf_int = np.exp(model.conf_int())
        
        result = {
            'test_type': 'Logistic Regression',
            'outcome_variable': outcome_var,
            'predictors': predictors,
            'n_observations': len(model_data),
            'model_summary': model.summary(),
            'odds_ratios': odds_ratios.to_dict(),
            'confidence_intervals': {
                var: [conf_int.loc[var, 0], conf_int.loc[var, 1]] 
                for var in conf_int.index
            },
            'p_values': model.pvalues.to_dict(),
            'pseudo_r_squared': model.prsquared,
            'aic': model.aic,
            'bic': model.bic
        }
        
        self.results[f'{outcome_var}_logistic_regression'] = result
        return result
    
    def correlation_with_significance(self, df: pd.DataFrame, method: str = 'pearson') -> Dict:
        """
        Calculate correlation matrix with significance testing.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if method == 'pearson':
            corr_matrix = numeric_df.corr(method='pearson')
            test_func = stats.pearsonr
        elif method == 'spearman':
            corr_matrix = numeric_df.corr(method='spearman')
            test_func = stats.spearmanr
        else:
            raise ValueError("Method must be 'pearson' or 'spearman'")
        
        # Calculate p-values
        p_matrix = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
        
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                if i != j:
                    try:
                        _, p_val = test_func(numeric_df.iloc[:, i], numeric_df.iloc[:, j])
                        p_matrix.iloc[i, j] = p_val
                    except:
                        p_matrix.iloc[i, j] = np.nan
                else:
                    p_matrix.iloc[i, j] = 0.0
        
        result = {
            'test_type': f'{method.capitalize()} Correlation',
            'correlation_matrix': corr_matrix,
            'p_value_matrix': p_matrix.astype(float),
            'significant_correlations': self._extract_significant_correlations(
                corr_matrix, p_matrix
            )
        }
        
        self.results[f'{method}_correlation'] = result
        return result
    
    def _extract_significant_correlations(self, corr_matrix: pd.DataFrame, 
                                        p_matrix: pd.DataFrame) -> List[Dict]:
        """
        Extract significant correlations from matrices.
        """
        significant = []
        
        for i in range(len(corr_matrix.index)):
            for j in range(i+1, len(corr_matrix.columns)):
                var1 = corr_matrix.index[i]
                var2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                p_val = p_matrix.iloc[i, j]
                
                if not np.isnan(p_val) and p_val < self.alpha:
                    significant.append({
                        'variable_1': var1,
                        'variable_2': var2,
                        'correlation': corr_val,
                        'p_value': p_val,
                        'strength': self._interpret_correlation_strength(abs(corr_val))
                    })
        
        return significant
    
    def _interpret_correlation_strength(self, abs_corr: float) -> str:
        """
        Interpret correlation strength based on absolute value.
        """
        if abs_corr < 0.1:
            return 'negligible'
        elif abs_corr < 0.3:
            return 'weak'
        elif abs_corr < 0.5:
            return 'moderate'
        elif abs_corr < 0.7:
            return 'strong'
        else:
            return 'very strong'
    
    def multiple_testing_correction(self, method: str = 'bonferroni') -> Dict:
        """
        Apply multiple testing correction to all stored p-values.
        """
        from statsmodels.stats.multitest import multipletests
        
        all_p_values = []
        test_names = []
        
        for test_name, result in self.results.items():
            if 'p_value' in result:
                all_p_values.append(result['p_value'])
                test_names.append(test_name)
            elif 'p_values' in result:
                for var, p_val in result['p_values'].items():
                    all_p_values.append(p_val)
                    test_names.append(f"{test_name}_{var}")
        
        if not all_p_values:
            return {'correction_applied': False, 'reason': 'No p-values found'}
        
        rejected, corrected_p_values, _, _ = multipletests(
            all_p_values, alpha=self.alpha, method=method
        )
        
        correction_results = {
            'correction_method': method,
            'original_alpha': self.alpha,
            'n_tests': len(all_p_values),
            'n_significant_original': sum(p < self.alpha for p in all_p_values),
            'n_significant_corrected': sum(rejected),
            'corrections': {
                test_names[i]: {
                    'original_p': all_p_values[i],
                    'corrected_p': corrected_p_values[i],
                    'significant_original': all_p_values[i] < self.alpha,
                    'significant_corrected': rejected[i]
                }
                for i in range(len(test_names))
            }
        }
        
        return correction_results
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive statistical analysis report.
        """
        report = []
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Significance level (alpha): {self.alpha}")
        report.append(f"Number of tests performed: {len(self.results)}")
        report.append("")
        
        for test_name, result in self.results.items():
            report.append(f"TEST: {test_name}")
            report.append("-" * 30)
            report.append(f"Type: {result['test_type']}")
            
            if 'p_value' in result:
                report.append(f"P-value: {result['p_value']:.6f}")
                report.append(f"Significant: {'Yes' if result['significant'] else 'No'}")
            
            if 'effect_size_cramers_v' in result:
                report.append(f"Effect size (Cramer's V): {result['effect_size_cramers_v']:.4f}")
            elif 'effect_size_r' in result:
                report.append(f"Effect size (r): {result['effect_size_r']:.4f}")
            elif 'effect_size_eta_squared' in result:
                report.append(f"Effect size (eta²): {result['effect_size_eta_squared']:.4f}")
            
            report.append("")
        
        return "\n".join(report)