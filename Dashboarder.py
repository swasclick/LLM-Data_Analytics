import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

path = str(input("Enter file path: "))
api_key = str(input("Enter api key: "))

# Helper Functions

def call_llm(prompt):
    # Define API endpoint and headers
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        # Make the API call
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error making API call: {e}")
        if e.response:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None

def load_csv(file_path, encodings=None):
   
    if encodings is None:
        encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            return df
        except Exception:
            pass

    # Return an error message if all encodings fail
    return "All attempts to load the file failed. Please check the file format and content."
    
# Preprocessing Functions

def findColumnType(df,col):
    cleaned = pd.Series(df[col].dropna().unique()).map(lambda x: x.lower() if isinstance(x, str) else x)
    
    # Check for ids
    isID = 'n'
    if 'id' in col.lower():
        if len(cleaned) < 0.6*len(df[col]):
            return df[col].name , 'id' , 'y' , 'y' , 'n' , 'n' 
        else:
            df = df.drop(col,axis=1) # drop if useless
    # check for boolean column
    if cleaned.isin([0,1,'0','1',True,False,'true','false','yes','no']).all():
        df[col] = df[col].map(lambda x: 
            x.lower() if isinstance(x, str) 
            else x).map({'0': 0,'1' : 1,'true': 1,True : 1,
                         'false': 0,False: 0,'yes': 1,'no': 0})
            
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
            return df[col].name , 'datetime' , grouped , 'n' , 'n' , 'y'
        return df[col].name , 'datetime' , grouped , 'n' , 'n' , 'y'
    column = cleaned.astype(str).str.strip()
    column = pd.to_datetime(column,errors='coerce')
    # Heuristic: if 80% valid values, parse into dates
    if column.notna().sum() / len(cleaned) > 0.8:
            df[col] = pd.to_datetime(df[col],errors='coerce')
            grouped = 'n'
            if len(cleaned) < 0.6*len(df[col].unique()):
                grouped = 'y'
                return df[col].name , 'datetime' , grouped , 'n' , 'n' , 'y'
            return df[col].name , 'datetime'  , grouped , 'n' , 'n', 'y'
    # Check if categorical
    if len(cleaned.astype(str)) < 20:
        return df[col].name , 'categorical', 'y' , 'y' , 'n' , 'n'
    # Else take it as unkown
    else:
        return df[col].name , 'unknown', 'y' , 'y' , 'n' , 'n'
    


# Plotting & Dashboarding Functions

# Running code
df = load_csv(path)

# Classify Columns
metadata = {}
for col in df.columns:
    column, colType, grouped, feature, sentiment, date = findColumnType(df,col)
    colData = {
        'type' : colType,
        'grouped': grouped,
        'feature' : feature,
        'sentiment': sentiment,
        'date' : date
    }
    metadata[col] = colData
    
# Use LLM to see what unknown columns are
sample_values = {}
for key,value in metadata.items():
    if metadata[key]['type'] == 'unknown':
        sample_values[key] = df[key][:20]


prompt = f"""
You are a highly reliable data science assistant. Your task is to classify data columns based on sample values. You will be provided a Python dictionary where:

- Keys are column names (strings)
- Values are a list of sample values from that column

Your job is to return a single Python dictionary where:
- Each key is the same column name
- Each value is a dictionary with the following **strict format**:

colData = {{
    "name": # The name of the column
    "type": "numeric" | "categorical" | "datetime" | "string" | "unknown",
    "grouped": true | false,        # Can we group/aggregate data by this column?
    "feature": true | false,        # Is this column suitable as a feature for machine learning modeling or trend analysis?
    "sentiment": true | false,      # Can we run sentiment analysis on this column? (is it a review?)
    "date": true | false            # Does this column contain dates or time info?
}}

Use only the allowed values above. Do not return any explanation or extra text.

Input:
{sample_values}

Output: Return ONLY the final Python dictionary described above, and nothing else.
"""

# prompt = f'''
# You are a data scientist assistant. Given the column values below, classify the type of data and return a 
# structured JSON with specific flags that help decide the next steps in data analysis.

# Column Sample Values:
# {sample_values}

# Return your answer as a Python dictionary in the following format **only**, using only the allowed values listed:

# ```python
# colData = {{
#     "name": # The name of the column
#     "type": "numeric" | "categorical" | "datetime" | "string" | "unknown",
#     "grouped": true | false,        # Can we group/aggregate data by this column?
#     "feature": true | false,        # Is this column suitable as a feature for machine learning modeling or trend analysis?
#     "sentiment": true | false,      # Can we run sentiment analysis on this column?
#     "date": true | false            # Does this column contain dates or time info?
# }}

# '''

print(call_llm(prompt))
print(metadata)