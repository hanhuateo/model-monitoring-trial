def object_to_category(df):
    nominal_features = df.drop(columns=['Attrition']).select_dtypes(include=['object']).columns.tolist()
    df[nominal_features] = df[nominal_features].astype('category')
    return df

def ordinal_encoding(df):
    # dictionary where the keys are the ordinal features, and values are the mapping from integers to categories
    ordinal_features_mapping = {'BusinessTravel': {1: 'Non-Travel', 2: 'Travel_Rarely', 3: 'Travel_Frequently'},
                            'Education': {1: 'Below College', 2: 'College', 3: 'Bachelor',  4: 'Master', 5: 'Doctor'},
                            'EnvironmentSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
                            'JobInvolvement': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
                            'JobLevel': {1: 'Junior', 2: 'Mid', 3: 'Senior', 4: 'Principal', 5: 'Head'},
                            'JobSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
                            'PerformanceRating': {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'},
                            'RelationshipSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
                            'StockOptionLevel': {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Very High'},
                            'WorkLifeBalance': {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
                            }
    ordinal_features = list(ordinal_features_mapping)
    