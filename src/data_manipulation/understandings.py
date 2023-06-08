def object_to_category(df):
    feature_names = df.drop(columns='Attrition').columns.to_list()
    nominal_features = df.drop(columns=['Attrition']).select_dtypes(include=['object']).columns.tolist()
    df[nominal_features] = df[nominal_features].astype('category')
    return df