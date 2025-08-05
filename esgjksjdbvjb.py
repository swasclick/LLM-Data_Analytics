import pandas as pd

df = pd.read_csv('Sample.csv')
def isdate(col):
            col2 = pd.to_datetime(col, errors='coerce',dayfirst=True)
            if col2.isnull().sum() > 0.5 * len(col):
                return False
            return True
print(pd.to_datetime(df['Dated'], errors='coerce',dayfirst=True))
print('yes' if isdate(df['Dated']) else 'no')