from .statstest import *
warnings.formatwarning = custom_warning

class StatsTest:
    
    cont_dtypes = ["int_", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float_", "float16", "float32", "float64"]
    cat_dtypes = ["category", "object"] 

    def __init__(self, data, alpha = 0.05):
        # input validation
        if not isinstance(data, pd.DataFrame) and not isinstance(data, np.ndarray):
            raise TypeError("Data must be in a pandas DataFrame or numpy array.")
        self.data = data
        # default level of significance (alpha) is 5%
        self.alpha = alpha

    def autostatstest(self, *columns, verbose = True):
        verbose = verbose
        num_cols = len(columns)
        # input validation
        if num_cols != 2:
            raise ValueError("Please enter 2 columns instead.")     
        for column in columns:
            if column not in self.data.columns:
                raise ValueError(f"Input column '{column}' is not in the dataset. Please try again.")
            if self.data[column].dtype not in (self.cont_dtypes + self.cat_dtypes):
                raise ValueError(f"Input column '{column}' is not of a valid datatype. Please try again.")
        if len(set(columns)) != len(columns):
            raise ValueError("Input columns must be unique. Please try again.")
        if self.data[columns[0]].dtype in self.cat_dtypes and self.data[columns[1]].dtype in self.cont_dtypes:
            raise ValueError("Logistic regression is recommended instead of a statistical test.")
        
        # normality test on continuous columns
        normality = {}
        for column in columns:
            if self.data[column].dtype in self.cont_dtypes:
                normal = normal_dist_test(self.data[column], alpha = self.alpha).normal
                normality[column] = normal
                if verbose:
                    if normal:
                        print(f"\nColumn '{column}' is normally distributed.")
                    else:
                        print(f"\nColumn '{column}' is not normally distributed.")

        # perform appropriate statistical test based on datatype of input columns
        results = {}
        if num_cols == 2:
            # categorical vs categorical
            if self.data[columns[0]].dtype in self.cat_dtypes and self.data[columns[1]].dtype in self.cat_dtypes:
                # concatenante columns into pandas dataframe
                df = pd.DataFrame({columns[0]: self.data[columns[0]], columns[1]: self.data[columns[1]]})
                # cross-tabulation
                observed_df = pd.crosstab(df[columns[0]], df[columns[1]])
                observed = observed_df.values
                # identify shape of contingency table
                fishers = False
                if observed_df.shape == (2,2):
                    fishers = True
                # check validity and conduct fishers exact test 
                if observed.min() < 5 and fishers:
                    if verbose:
                        print("\nFisher's Exact Test is recommended as assumption of Chi-square Test of Independence is violated (Each cell in contingency table should be at least 5) and contingency table is 2x2 in shape.")
                    result = fishers_test(observed_df, self.alpha, verbose = verbose)
                    return result
                # else conduct chi2 test
                result = chisquare_test(observed_df, self.alpha, verbose = verbose)
                results["chi2"] = result    
                # carry out post-hoc analysis using multiple comparisons correction if chi2 test is significant
                if result.significant:
                    # check that independent variable has more than 2 categories
                    if self.data[columns[1]].nunique() > 2:
                        try:
                            result = chisquare_posthoc_test(observed_df, self.alpha, verbose = verbose)
                            results["post-hoc"] = result
                        except ValueError:
                            warnings.warn("Post-hoc analysis cannot be conducted as internally computed table of expected frequencies has zero elements.")
                else:
                    return result

            # continuous vs continuous 
            elif self.data[columns[0]].dtype in self.cont_dtypes and self.data[columns[1]].dtype in self.cont_dtypes:
                # conduct pearson's correlation if both columns have normal distribution
                if normality[columns[0]] and normality[columns[1]]:
                    result = pearsonr_test(self.data[columns[0]], self.data[columns[1]], self.alpha, verbose = verbose)
                    return result
                # conduct spearman's correlation if one or both columns have non-normal distribution
                else:
                    if verbose:
                        print("\nNon-parametric Spearman's Correlation Test is recommended as one or more columns have non-normal distribution.")
                    result = spearmanr_test(self.data[columns[0]], self.data[columns[1]], self.alpha, verbose = verbose)
                    return result

            # continuous vs categorical
            elif self.data[columns[0]].dtype in self.cont_dtypes and self.data[columns[1]].dtype in self.cat_dtypes:
                # check for number of unique categories of independent variable 
                if self.data[columns[1]].nunique() == 2:
                    # group samples by category
                    sample1 = self.data[self.data[columns[1]] == self.data[columns[1]].unique()[0]][columns[0]]
                    sample2 = self.data[self.data[columns[1]] == self.data[columns[1]].unique()[1]][columns[0]]
                    # conduct student's t-test if dependent variable is normally distributed
                    if normality[columns[0]]:
                        # test for equal variance
                        equal_var = equal_variance_test(sample1, sample2, normal = normality[columns[0]], alpha = self.alpha).equal_variance
                        result = t_test(sample1, sample2, alpha = self.alpha, equal_variance = equal_var, verbose = verbose)
                        return result
                    else:
                        # conduct mann-whitney u test if dependent variable is not normally distributed
                        if verbose:
                            print("\nNon-parametric Mann-Whitney U Test is recommended as dependent variable has non-normal distribution.")
                        result = mannwhitneyu_test(sample1, sample2, alpha = self.alpha, verbose = verbose)
                        return result

                elif self.data[columns[1]].nunique() > 2:
                    # group samples by category 
                    samples = []
                    for category in self.data[columns[1]].unique():
                        samples.append((self.data[self.data[columns[1]] == category][columns[0]]).tolist())
                    # test for equal variance between samples
                    equal_var = equal_variance_test(*samples, normal = normality[columns[0]], alpha = self.alpha).equal_variance
                    # conduct one-way ANOVA test if dependent variable is normally distributed
                    if normality[columns[0]]: 
                        if verbose:
                            print("\nOne-way ANOVA is recommended as independent variable has more than 2 categories.")
                            if not equal_var:
                                warnings.warn("Assumption of One-way ANOVA is violated (Samples should have equal variance). Results may be inaccurate.")
                        result = onewayanova_test(*samples, alpha = self.alpha, equal_variance = equal_var, verbose = verbose)
                        results["anova"] = result
                        # conduct post-hoc analysis using tukey's HSD test if one-way ANOVA test is significant
                        if result.significant:
                            df = pd.DataFrame({columns[0]: self.data[columns[0]], columns[1]: self.data[columns[1]]})
                            result = tukeys_test(df, columns[0], columns[1], alpha = self.alpha, verbose = verbose)
                            results["post-hoc"] = result
                        else:
                            return result
        
                    # conduct kruskal-wallis test if dependent variable is not normally distributed
                    else:
                        if verbose:
                            print("\nNon-parametric Kruskal-Wallis Test is recommended as dependent variable has non-normal distribution.")
                            if not equal_var:
                                warnings.warn("Assumption of Kruskal-Wallis Test is violated (Samples should have equal variance). Results may be inaccurate.")
                        result = kruskalwallis_test(*samples, alpha = self.alpha, equal_variance = equal_var, verbose = verbose)
                        results["kw"] = result
                        # conduct post-hoc analysis using dunn's test if kruskal-wallis test is significant
                        if result.significant:
                            df = pd.DataFrame({columns[0]: self.data[columns[0]], columns[1]: self.data[columns[1]]})
                            result = dunns_test(df, columns[0], columns[1], alpha = self.alpha, verbose = verbose)
                            results["post-hoc"] = result
                        else:
                            return result

        return results
    
    def autostatstest_all(self, dependent_col, verbose = True):
        verbose = verbose
        if dependent_col not in self.data.columns:
            raise ValueError(f"Input column '{dependent_col}' is not in the dataset. Please try again.")
        if self.data[dependent_col].dtype not in (self.cont_dtypes + self.cat_dtypes):
            raise ValueError(f"Input column '{dependent_col}' is not of a valid datatype. Please try again.")
        
        stored_warnings = []
        results = {}
        count = 1
        for column in self.data.columns:
            if column != dependent_col:
                if verbose:
                    print(f"\n\033[1mTest {count}: {dependent_col} vs {column}\033[0m")
                count += 1
                try:
                    with warnings.catch_warnings(record = True) as w:
                        result = self.autostatstest(dependent_col, column, verbose = verbose)
                        if w:
                            stored_warnings.extend([str(warning.message) for warning in w])
                            print()
                            if verbose:
                                print(stored_warnings[-1])
                    results[column] = result
                except (ValueError, TypeError) as e:
                    if verbose:
                        print("\n" + str(e))
                    continue
                
        return results