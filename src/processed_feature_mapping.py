def mapping(df, column_transformer):

    # df.columns = ['Age', 'BusinessTravel', 'DailyRate', 'Department_Human Resources', 
    #               'Department_Research & Development', 'Department_Sales', 'DistanceFromHome', 'Education', 
    #               'EducationField_Human Resources', 'EducationField_Life Sciences', 'EducationField_Marketing', 
    #               'EducationField_Medical', 'EducationField_Other', 'EducationField_Technical Degree', 
    #               'Environment Satisfaction', 'Gender_Female', 'Gender_Male', 'HourlyRate', 'JobInvolvement', 
    #               'JobLevel', 'JobRole_Healthcare Representative', 'JobRole_Human Resources', 
    #               'JobRole_Laboratory Technician', 'JobRole_Manager', 'JobRole_Manufacturing Director', 
    #               'JobRole_Research Director', 'JobRole_Research Scientist', 'JobRole_Sales Executive', 
    #               'JobRole_Sales Representative', 'JobSatisfaction', 'MaritalStatus_Divorced', 
    #               'MaritalStatus_Married', 'MaritalStatus_Single', 'MonthlyIncome', 'MonthlyRate',
    #               'NumCompaniesWorked', 'OverTime_No', 'OverTime_Yes', 'PercentSalaryHike', 'PerformanceRating', 
    #               'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
    #               'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSincePromotion',
    #               'YearsWithCurrManager']
    df.columns = [column_transformer.get_feature_names_out()]
    return df