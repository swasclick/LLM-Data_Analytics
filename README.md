# Autolysis: Automated CSV Data Cleaning, Analysis & Insights

This Python project, autolysis.py, offers a streamlined approach to cleaning, processing, and analyzing data from CSV files. It empowers users to leverage the power of Large Language Models (LLMs) for enhanced data exploration and understanding.

## Key Features:

- ### Effortless Data Cleaning:  
  Missing value imputation  
  Data type detection  
  Inconsistency removal and more  

- ### Comprehensive Analysis:  
  Descriptive statistics  
  Data visualizations (matplotlib or seaborn)  

- ### LLM-powered Insights:  
  Generate meaningful summaries and descriptions  
  Identify patterns and trends  
  Formulate insightful hypothesis and then answer them using surrounding data  

## Getting Started:

Prerequisites: Ensure you have Python installed on your system.

Use the [uv package manager](https://www.youtube.com/watch?v=igWlYl3asKw&t=1240s) to run the script via a command line argument (in the script folder) of the form:  

   ```python
      uv run autolysis.py <dataset_path>.csv
```

## How It Works:

- **Data Loading:** The autolysis.py script begins by loading the CSV file specified by the user.  
- **Data Cleaning:** The script applies various cleaning techniques to ensure data quality and consistency.  
- **LLM Processing:** The script leverages an LLM to gain deeper insights from the data like catagorising the data's variables or giving summary statistics.  
- **Analysis and Visualization:** The script performs analytical tasks (Clustering, Principal Component Analysis, sampling for large datasets) and generates informative data visualizations.  

## Disclaimer: The performance and accuracy of LLM-generated insights may vary depending on data quality and the specific LLM model used.
