import pandas as pd
import featuretools as ft
from woodwork.logical_types import Categorical

df = pd.read_csv("retail.csv", parse_dates=["invoice_date"])

df.head()

es = ft.EntitySet(id="data")

es = es.add_dataframe(
    dataframe=df,              
    dataframe_name="data",     
    index="rows",              
    make_index=True,           
    time_index="invoice_date", 
    logical_types={
        "customer_id": Categorical, 
        "invoice": Categorical,
    },
)

es["data"].ww

es.normalize_dataframe(
    base_dataframe_name="data",    
    new_dataframe_name="invoices", 
    index="invoice",               
    copy_columns=["customer_id"],   
)

es["data"].head()

cum_primitives = ["cum_sum", "cum_max", "diff", "time_since_previous"]

general_primitives = ["sine", "cosine"]

feature_defs = ft.dfs(
    entityset=es,                              
    target_dataframe_name="data",              
    agg_primitives=[],                          
    trans_primitives=general_primitives,         
    groupby_trans_primitives = cum_primitives,   
    ignore_dataframes = ["invoices"],            
    features_only=True,    
)

feature_matrix, feature_defs = ft.dfs(
    entityset=es,                                
    target_dataframe_name="data",                
    agg_primitives=[],                           
    trans_primitives=general_primitives,         
    groupby_trans_primitives = cum_primitives,   
    ignore_dataframes = ["invoices"],            
)

feature_matrix.head()

feature_matrix.shape

import numpy as np
import pandas as pd

df = pd.read_csv("retail.csv", parse_dates=["invoice_date"])

df.head()

numeric_vars = ["quantity", "price"]

func = ["cumsum", "cummax", "diff"]

new_names = [f"{var}_{function}" for function in func for var in numeric_vars]

df[new_names] = df.groupby("invoice")[numeric_vars].agg(func)

df[df["invoice"] == "489434" ][numeric_vars + new_names].head()

new_names = [f"{var}_{function}" for function in ["sin", "cos"]for var in numeric_vars]

df[new_names] = df[numeric_vars].agg([np.sin, np.cos])
df[new_names].head()

