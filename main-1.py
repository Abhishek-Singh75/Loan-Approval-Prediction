import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv('loan_prediction.csv')
print(df)
#Generate Report
profile = ProfileReport(df, title="Loan Approval prediction")
profile.to_file("loan Prediction.html")