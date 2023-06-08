def object_to_category(df):
    nominal_features = df.drop(columns=['Attrition']).select_dtypes(include=['object']).columns.tolist()
    df[nominal_features] = df[nominal_features].astype('category')
    return df