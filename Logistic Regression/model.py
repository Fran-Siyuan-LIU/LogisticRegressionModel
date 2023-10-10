from LR import LogisticRegression

import numpy as np
import pandas as pd

filename = "Loan_Data.csv"

df = pd.read_csv(filename)

cols = list(df.columns)

for col in cols:
    print(col)


'''
Loan_ID
Gender
Married
Dependents
Education
Self_Employed
ApplicantIncome
CoapplicantIncome
LoanAmount
Loan_Amount_Term
Credit_History
Property_Area
Loan_Status
'''