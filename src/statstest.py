import pandas as pd
import numpy as np
import warnings
from scipy.stats import normaltest, levene, ttest_ind, mannwhitneyu, f_oneway, kruskal, pearsonr, spearmanr, chi2_contingency, fisher_exact
from scikit_posthocs import posthoc_tukey, posthoc_dunn   
from statsmodels.sandbox.stats.multicomp import multipletests
from itertools import combinations
from .stats_classes import *

def print_verbose(test_conducted, result, null_hypo, additional_info, alpha):
    print(f"\n----- Conducted {test_conducted} -----\n")
    if additional_info:
        for info in additional_info:
            print(info)
        print()
    print(f"{test_conducted} statistic = {result.statistic:.06f}, p-value = {result.pvalue:.06f}")
    print("Level of significance = " + str(alpha))
    print("Null hypothesis (H0) = " + null_hypo)
    print(f"Conclusion = {'Reject' if result.pvalue <= alpha else 'Fail to reject'} null hypothesis (H0)")

def custom_warning(message, category, filename, lineno, line = None):
    return f"\n{filename}:{lineno}: {category.__name__}: {message}\n"

def normal_dist_test(arr, alpha):
    # input validation
    if not isinstance(arr, np.ndarray) and not isinstance(arr, pd.Series):
        raise TypeError("Input must be in a pandas Series or numpy array.")
    # conduct d'agostino k2 test 
    result = normaltest(arr)
    normal = True if result.pvalue > alpha else False
    return DatkNormResult(normal = normal, pvalue = result.pvalue)

def equal_variance_test(*args, normal, alpha):
    # input validation
    if len(args) == 0 or len(args) == 1:
        raise ValueError("Please provide at least 2 arguments.")
    valid = (np.ndarray, pd.Series, list)
    for arg in args:
        if not isinstance(arg, valid):
            raise TypeError("Input must be in a pandas Series, numpy array, or list.")
    center = "mean" if normal else "median"
    # conduct levene's test
    result = levene(*args, center = center)
    equal_variance = True if result.pvalue > alpha else False
    return DatkEqualVarResult(pvalue = result.pvalue, equal_variance = equal_variance)

def t_test(sample1, sample2, alpha, equal_variance, verbose = True):
    # input validation
    if not isinstance(sample1, np.ndarray) and not isinstance(sample1, pd.Series):
        raise TypeError("Sample 1 must be in a pandas Series or numpy array.")
    if not isinstance(sample2, np.ndarray) and not isinstance(sample2, pd.Series):
        raise TypeError("Sample 2 must be in a pandas Series or numpy array.")
    # conduct student's independent t-test
    result = ttest_ind(sample1, sample2, equal_var = equal_variance)
    sig = True if result.pvalue <= alpha else False
    if verbose:
        sample1_mean = round(sample1.mean(),4)
        sample2_mean = round(sample2.mean(),4)
        test_conducted = "Independent Student's T-Test" if equal_variance else "Welch's T-test"
        additional_info = [f"Samples have {'equal' if equal_variance else 'non-equal'} variance according to Levene Test", f"Sample 1 mean: {sample1_mean}", f"Sample 2 mean: {sample2_mean}"]
        print_verbose(test_conducted = test_conducted, result = result, null_hypo = "Samples have identical means", additional_info = additional_info, alpha = alpha)
    return DatkTTestResult(significant = sig, pvalue = result.pvalue, statistic = result.statistic)

def mannwhitneyu_test(sample1, sample2, alpha, verbose = True):
    # input validation
    if not isinstance(sample1, np.ndarray) and not isinstance(sample1, pd.Series):
        raise TypeError("Sample 1 must be in a pandas Series or numpy array.")
    if not isinstance(sample2, np.ndarray) and not isinstance(sample2, pd.Series):
        raise TypeError("Sample 2 must be in a pandas Series or numpy array.")
    # conduct mann-whitney u test
    result = mannwhitneyu(sample1, sample2)
    sig = True if result.pvalue <= alpha else False
    if verbose: 
        sample1_median = round(sample1.median(),4)
        sample2_median = round(sample2.median(),4)
        additional_info = [f"Sample 1 median: {sample1_median}", f"Sample 2 median: {sample2_median}"]
        print_verbose(test_conducted = "Mann-Whitney U Test", result = result, null_hypo = "Samples have identical medians", additional_info = additional_info, alpha = alpha)
    return DatkMannWhitneyUResult(significant = sig, pvalue = result.pvalue, statistic = result.statistic)

def onewayanova_test(*samples, alpha, equal_variance, verbose = True):
    # input validation
    valid = (np.ndarray, pd.Series, list)
    for sample in samples:
        if not isinstance(sample, valid):
            raise TypeError("Sample must be in a pandas Series or numpy array.")
    # conduct one-way anova
    result = f_oneway(*samples)
    sig = True if result.pvalue <= alpha else False
    if verbose:
        additional_info = [f"Samples have {'equal' if equal_variance else 'non-equal'} variance according to Levene Test"]      
        for i, sample in enumerate(samples):
            sample_mean = sum(sample)/len(sample)
            additional_info.append(f"Sample {i+1} mean: {sample_mean}")
        print_verbose(test_conducted = "One-way ANOVA", result = result, null_hypo = "Samples have identical means", additional_info = additional_info, alpha = alpha)
    return DatkOneWayAnovaResult(significant = sig, pvalue = result.pvalue, statistic = result.statistic)

def kruskalwallis_test(*samples, alpha, equal_variance, verbose = True):
    # input validation
    valid = (np.ndarray, pd.Series, list)
    for sample in samples:
        if not isinstance(sample, valid):
            raise TypeError("Sample must be in a pandas Series or numpy array.")
    # conduct kruskal-wallis test
    result = kruskal(*samples)
    sig = True if result.pvalue <= alpha else False
    if verbose:
        additional_info = [f"Samples have {'equal' if equal_variance else 'non-equal'} variance according to Levene Test"]
        for i, sample in enumerate(samples):
            sample_median = np.median(sample)
            additional_info.append(f"Sample {i+1} median: {sample_median}")
        print_verbose(test_conducted = "Kruskal-Wallis Test", result = result, null_hypo = "Samples have identical medians", additional_info = additional_info, alpha = alpha)
    return DatkKruskalWallisResult(significant = sig, pvalue = result.pvalue, statistic = result.statistic)

def chisquare_test(observed, alpha, verbose = True):
    # input validation 
    if not isinstance(observed, pd.DataFrame) and not isinstance(observed, np.ndarray):
        raise TypeError("Input must be in a pandas DataFrame or numpy array.")
    observed_df = observed.to_markdown()
    # check validity of chi2
    if observed.values.min() < 5:
        warnings.warn("Assumption of Chi-square Test of Independence is violated (Each cell in contingency table should be at least 5).")
    # conduct chi2 test
    result = chi2_contingency(observed)
    sig = True if result.pvalue <= alpha else False
    if verbose:
        print_verbose(test_conducted = "Chi-square Test of Independence", result = result, null_hypo = "Variables are independent from each other", additional_info = [observed_df], alpha = alpha)
    return DatkChi2Result(significant = sig, pvalue = result.pvalue, statistic = result.statistic)

def fishers_test(observed, alpha, verbose= True):
    # input validation 
    if not isinstance(observed, pd.DataFrame) and not isinstance(observed, np.ndarray):
        raise TypeError("Input must be in a pandas DataFrame or numpy array.")
    observed_df = observed.to_markdown()
    # check shape of contingency table
    if observed.shape != (2,2):
        raise ValueError("Contingency table must be 2x2 in shape.")
    # conduct fisher's exact test
    result = fisher_exact(observed)
    sig = True if result.pvalue <= alpha else False
    if verbose:
        print_verbose(test_conducted = "Fisher's Exact Test", result = result, null_hypo = "Variables are independent from each other", additional_info = [observed_df], alpha = alpha)
    return DatkFishersTestResult(significant = sig, pvalue = result.pvalue, statistic = result.statistic)
        
def pearsonr_test(var1, var2, alpha, verbose = True):
    # input validation
    if not isinstance(var1, pd.Series) and not isinstance(var1, np.ndarray):
        raise TypeError("Column 1 must be in a pandas Series or numpy array.")
    if not isinstance(var2, pd.Series) and not isinstance(var2, np.ndarray):
        raise TypeError("Column 2 must be in a pandas Series or numpy array.")
    # check for assumption of related pairs: each observation should have a pair of values
    if len(var1) != len(var2):
        raise ValueError("Columns must have the same length (paired observations).")
    # conduct pearson's correlation
    result = pearsonr(var1, var2)
    sig = True if result.pvalue <= alpha else False
    if verbose:
        print_verbose(test_conducted = "Pearson's Correlation", result = result, null_hypo = "No linear relationship between the variables", additional_info = [], alpha = alpha)
    return DatkPearsonrResult(significant = sig, pvalue = result.pvalue, statistic = result.statistic)

def spearmanr_test(var1, var2, alpha, verbose = True):
    # input validation
    if not isinstance(var1, pd.Series) and not isinstance(var1, np.ndarray):
        raise TypeError("Column 1 must be in a pandas Series or numpy array.")
    if not isinstance(var2, pd.Series) and not isinstance(var2, np.ndarray):
        raise TypeError("Column 2 must be in a pandas Series or numpy array.")
    # check for assumption of related pairs: each observation should have a pair of values
    if len(var1) != len(var2):
        raise ValueError("Columns must have the same length (paired observations).")
    # conduct spearman's correlation
    result = spearmanr(var1, var2)
    sig = True if result.pvalue <= alpha else False
    if verbose:
        print_verbose(test_conducted = "Spearman's Correlation", result = result, null_hypo = "No monotonic relationship between the variables", additional_info = [], alpha = alpha)
    return DatkSpearmanrResult(significant = sig, pvalue = result.pvalue, statistic = result.statistic)

# post-hoc tests

def tukeys_test(df, col1, col2, alpha, verbose = True):
    # input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be in a pandas DataFrame.")
    if verbose:
        print("\n----- Conducted post-hoc analysis using Tukey's HSD Test -----\n")
    result = posthoc_tukey(df, val_col = col1, group_col = col2)
    num_categories = result.shape[0]
    results = []
    for i in range(num_categories):
        for j in range(i+1, num_categories):
            pair = (result.columns[i], result.columns[j])
            pvalue = result.iloc[i,j]
            sig = True if pvalue <= alpha else False
            if verbose:
                if pvalue <= alpha:
                    print(f"{pair}: p-value = {pvalue:}, significant = {sig}")
                else:
                    print(f"{pair}: p-value = {pvalue:}, significant = {sig}")
            result2 = DatkPostHocResult(combi = pair, pvalue = pvalue, significant = sig)
            results.append(result2)
    return results

def dunns_test(df, col1, col2, alpha, verbose = True):
    # input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be in a pandas DataFrame.")
    if verbose:
        print("\n----- Conducted post-hoc analysis using Dunn's Test -----\n")
    result = posthoc_dunn(df, val_col = col1, group_col = col2)
    num_categories = result.shape[0]
    results = []
    for i in range(num_categories):
        for j in range(i+1, num_categories):
            pair = (result.columns[i], result.columns[j])
            pvalue = result.iloc[i,j]
            sig = True if pvalue <= alpha else False
            if verbose:
                if pvalue <= alpha:
                    print(f"{pair}: p-value = {pvalue:}, significant = {sig}")
                else:
                    print(f"{pair}: p-value = {pvalue:}, significant = {sig}")
            result2 = DatkPostHocResult(combi = pair, pvalue = pvalue, significant = sig)
            results.append(result2) 
    return results

# https://neuhofmo.github.io/chi-square-and-post-hoc-in-python/

def chisquare_posthoc_test(df, alpha, verbose = True):
    # input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be in a pandas DataFrame.")
    # gather all combinations for the post-hoc test 
    all_combinations = list(combinations(df.columns, 2))
    p_values = []
    results = []
    for comb in all_combinations:
        new_df = df[[comb[0], comb[1]]]
        result = chi2_contingency(new_df)
        p_values.append(result.pvalue)
    significant_list, corrected_p_values = multipletests(p_values, alpha = alpha, method = "fdr_bh")[:2]
    if verbose:
        print("\n----- Conducted Post-hoc Analysis using Multiple Comparisons Correction -----\n")
    for p_val, corr_p_val, sig, comb in zip(p_values, corrected_p_values, significant_list, all_combinations):
        if verbose:
            print(f"{comb}: p-value = {p_val}, corrected p-value = {corr_p_val}, significant = {sig}")
        result = DatkChi2PostHocResult(combi = comb, pvalue = p_val, corrected_pvalue = corr_p_val, significant = sig)
        results.append(result)
    return results

# user functions

def print_significant_categories(result):
    results = result["post-hoc"]
    for result in results:
        if result.significant:
            print(result.combi)