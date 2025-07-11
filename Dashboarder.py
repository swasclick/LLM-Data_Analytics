import pandas as pd
import numpy as np
#import plotly.graph_objects as go

# Helper Functions
# def call_llm(prompt):

# def openFile():
    
# Preprocessing Functions

def findColumnType(df,col):
    cleaned = df[col].dropna().unique().map(lambda x: x.lower() if isinstance(x, str) else x)
    
    # Check for ids
    isID = 'n'
    if 'id' in col.lower():
        if len(cleaned) < 0.6*len(df[col]):
            return df[col].name , 'id' , 'y' , 'y' , 'n' , 'n' 
        else:
            df = df.drop(col,axis=1) # drop if useless
    # check for boolean column
    if cleaned.isin([0,1,'0','1',True,False,'true','false','yes','no']).all():
        df[col] = df[col].map(lambda x: x.lower() if isinstance(x, str) else x).map({'0': 0,'1' : 1,'true': 1,True : 1,'false': 0,False: 0,'yes': 1,'no': 0})
        return df[col].name , 'boolean' , 'y' , 'y' , 'n' , 'n' 
    # check for numerical column
    if pd.to_numeric(df[col], errors='coerce').notna().all():
        df[col] = pd.to_numeric(df[col], errors='coerce')
        grouped = 'n'
        if len(df[col].unique()) < 10:
            grouped = 'y'
        return  df[col].name , 'categorical' , grouped , 'y' , 'n' , 'n' 
    # Check for datetime column
    if pd.api.types.is_datetime64_any_dtype(cleaned):
        grouped = 'n'
        if len(cleaned) < 0.6*len(df[col].unique()):
            grouped = 'y'
            return df[col].name , 'datetime' , 'n' , grouped , 'n' , 'n' , 'y'
        return df[col].name , 'datetime' , 'n' , grouped , 'n' , 'n' , 'y'
    column = cleaned.astype(str).str.strip()
    column = pd.to_datetime(column,errors='coerce')
    # Heuristic: if 80% valid values, parse into dates
    if column.notna().sum() / len(cleaned) > 0.8:
            df[col] = pd.to_datetime(df[col],errors='coerce')
            grouped = 'n'
            if len(cleaned) < 0.6*len(df[col].unique()):
                grouped = 'y'
                return df[col].name , 'datetime' , 'n' , grouped , 'n' , 'n' , 'y'
            return df[col].name , 'datetime' , 'n' , grouped , 'n' , 'n' , 'y'
    # Check if categorical
    if len(cleaned.astype(str)) < 15:
        return df[col] , 

# Plotting & Dashboarding Functions