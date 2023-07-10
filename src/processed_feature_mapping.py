def mapping(df):
    df.columns = ['Age', 'BusinessTravel', 'DailyRate', 'Human Resources', 'Research & Development', 
                  'Sales', 'DistanceFromHome', 'Education', 'Human Resources', 'Life Sciences',
                  'Marketing', 'Medical', 'Other', 'Technical Degree', 'Environment Satisfaction', 
                  'Female', 'Male', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'Healthcare Representative',
                  'Human Resources', 'Laboratory Technician', 'Manager', ' Manufacturing Director', 
                  'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative',
                  'JobSatisfaction', 'Divorced', 'Married', 'Single', 'MonthlyIncome', 'MonthlyRate',
                  'NumCompaniesWorked', 'No', 'Yes', 'PercentSalaryHike', 'PerformanceRating', 
                  'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                  'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSincePromotion',
                  'YearsWithCurrManager']
    return df