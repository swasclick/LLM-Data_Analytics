# Ensure dates have a format that starts with the day, not month or year
import pandas as pd
import numpy as np
import requests, json
from pprint import pprint
import plotly.graph_objects as go
import os
from openai import OpenAI
from together import Together

path = str(input("Enter file path: "))
api_key = str(input("Enter api key: "))

# Prompts used
sample_values = {}
classification_prompt = f"""
You are a highly reliable data science assistant. Your task is to classify data columns based on sample values. 
You will be provided a Python dictionary where:

- Keys are column names (strings)
- Values are a list of sample values from that column

Your job is to return a single Python dictionary where:
- Each key is the same column name
- Each value is a dictionary with the following **strict format**:

colData = {{
    "type": "numeric" | "categorical" | "datetime" | "string" | "unknown",
    "grouped": "true" | "false",        # Can we group/aggregate data by this column?
    "feature": "true" | "false",        # Is this column suitable as a feature for machine learning modeling or trend analysis?
    "sentiment": "true" | "false",      # Can we run sentiment analysis on this column? (is it a review?)
    "date": "true" | "false"            # Does this column contain dates or time info?
}}

Use only the allowed values above. Do not return any explanation or extra text.

Input:
{sample_values}

Output: Return ONLY the final Python dictionary described above, and nothing else.
"""


# Helper Functions

def call_llm(prompt):
    # try:
    #     client = OpenAI(
    #         base_url="https://router.huggingface.co/v1",
    #         api_key=api_key,
    #     )

    #     completion = client.chat.completions.create(
    #         model="zai-org/GLM-4.5:novita",
    #         messages=[
    #             {"role": "system", "content": "You are a data analyst. Respond with only the final answer, without prefacing it with any thinking or explanation."},
    #             {"role": "user", "content": prompt + " Only return the final JSON dictionary output. Do not include <think> blocks or internal reasoning."}

    #         ],
    #     )
    #     print((completion.choices[0].message.content))
    #     return completion.choices[0].message.content
    
    # except Exception as e:
    #     # Print any error: could be network, bad key, model error, etc.
    #     print(f"Error occurred: {e}")
    # return None

    # client = Together(api_key=api_key)
    # try:
    #     # Make the API call
    #     print("> CALLING LLM...\n")
    #     # response = requests.post(url, headers=headers, json=data)
    #     response = client.chat.completions.create(
    #     model="meta-llama/Llama-Vision-Free",
    #     messages=[
    #     {
    #         "role": "user",
    #         "content": "What are some fun things to do in New York?"
    #     }
    #     ]
    # )
    #     response.raise_for_status()  # Raise an exception for HTTP errors
    #     return response.json()["choices"][0]["message"]["content"]
    # except requests.exceptions.RequestException as e:
    #     print(f"Error making API call: {e}")
    #     if e.response:
    #         print(f"Response status code: {e.response.status_code}")
    #         print(f"Response text: {e.response.text}")
    #     return None
    
    client = Together(api_key=api_key)

    try:
        print("> CALLING LLM...\n")

        response = client.chat.completions.create(
            model="Qwen/Qwen2-72B-Instruct",  # Or: meta-llama/Llama-3-70B-Instruct
            messages=[
                {
                    "role": "user",
                    "content": f'{prompt} Only return the final JSON dictionary output. Do not include <think> blocks or internal reasoning.'
                }
            ],
            max_tokens=256,
            temperature=0.7
        )

        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error occurred: {e}")
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
    print("All attempts to load the file failed. Please check the file format and content.") 
    
# Preprocessing & EDA Functions

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
    
    def isdate(col):
            col2 = pd.to_datetime(col, errors='coerce',dayfirst=True)
            if col2.isnull().sum() > 0.5 * len(col):
                return False
            return True

    if isdate(cleaned): #check if it is date type column
        grouped = 'n'
        if len(cleaned) > 100:
            grouped = 'y'
        return df[col].name , 'datetime' , grouped , 'n' , 'n' , 'y'
    
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

if sample_values:
    print(f'\n--- THE COLUMNS HAVE BEEN IDENTIFIED AS FOLLOWS---\n')
    pprint(metadata) 
    print()   
    response = call_llm(classification_prompt)
    if isinstance(response,str):
        response = json.loads(response)
    if response:
        for key,value in response.items():
            for k,v in value.items():
                if value[k] == 'true':
                    value[k] = 'y'
                elif value[k] == 'false':
                    value[k] = 'n'
            metadata[key] = value

print(f'--- THE COLUMNS HAVE BEEN CLASSIFIED AS FOLLOWS---\n')
pprint(metadata)
print()

# Perform EDA

'''dtypes : boolen, numerical,datetime,categorical,string'''
'''yes/no: grouping, ml features, sentiment, time series

- Numerical: histogram, boxplot, scatterplot
- Categorical: bar chart, pie chart
- Boolean: pie chart
- Datetime: time series plot, histogram
- String: bar chart, word cloud
- Unknown: bar chart, pie chart
'''

# Boolean Columns
bool_cols = [col for col in df.columns if metadata[col]['type'] == 'boolean']
if bool_cols:
    print(f'--- BOOLEAN COLUMNS ---\n{bool_cols}\n')    
    for col in bool_cols:
        fig = go.Figure(data=[go.Pie(labels=df[col].value_counts().index, values=df[col].value_counts().values)])
        fig.update_layout(title_text=f'Boolean Column: {col}')
        fig.show()
        
        
# Numerical Columns
num_cols = [col for col in df.columns if metadata[col]['type'] == 'numeric']
if num_cols:
    print(f'--- NUMERICAL COLUMNS ---\n{num_cols}\n')
    for col in num_cols:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[col], nbinsx=50))
        fig.update_layout(title_text=f'Numerical Column: {col}', xaxis_title=col, yaxis_title='Count')
        fig.show()
        
# Categorical Columns
cat_cols = [col for col in df.columns if metadata[col]['type'] == 'categorical']
if cat_cols:
    print(f'--- CATEGORICAL COLUMNS ---\n{cat_cols}\n')
    for col in cat_cols:
        fig = go.Figure(data=[go.Bar(x=df[col].value_counts().index, y=df[col].value_counts().values)])
        fig.update_layout(title_text=f'Categorical Column: {col}', xaxis_title=col, yaxis_title='Count')
        fig.show()
        
# Datetime Columns
date_cols = [col for col in df.columns if metadata[col]['type'] == 'datetime']
if date_cols:
    print(f'--- DATETIME COLUMNS ---\n{date_cols}\n')
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce',dayfirst=True)
        fig = go.Figure()
        column = cat_cols[0]
        fig.add_trace(go.Scatter(x=df[col], y=df[column], mode='lines+markers', name='Time Series'))
        fig.update_layout(title_text=f'Datetime Column: {col}', xaxis_title=col, yaxis_title='Count')
        fig.show()
# String Columns
str_cols = [col for col in df.columns if metadata[col]['type'] == 'string']
if str_cols:
    print(f'--- STRING COLUMNS ---\n{str_cols}\n')
    for col in str_cols:
        fig = go.Figure(data=[go.Bar(x=df[col].value_counts().index, y=df[col].value_counts().values)])
        fig.update_layout(title_text=f'String Column: {col}', xaxis_title=col, yaxis_title='Count')
        fig.show()
        