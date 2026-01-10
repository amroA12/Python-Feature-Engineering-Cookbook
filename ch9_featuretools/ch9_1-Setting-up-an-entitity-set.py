import pandas as pd
import featuretools as ft
from woodwork.logical_types import Categorical

df = pd.read_csv("retail.csv", parse_dates=["invoice_date"])

df.head()

df.tail()
df["customer_id"].nunique()
df["invoice"].nunique() 
df["stock_code"].nunique()
len(df)

es = ft.EntitySet(id="data")

es = es.add_dataframe(
    dataframe=df,              
    dataframe_name="data",     
    index="rows",              
    make_index=True,           
    time_index="invoice_date",
    logical_types={
        "customer_id": Categorical, 
    },
)

es.normalize_dataframe(
    base_dataframe_name="data",     
    new_dataframe_name="invoices",  
    index="invoice",                
    copy_columns=["customer_id"],   
)

es.normalize_dataframe(
    base_dataframe_name="invoices",  
    new_dataframe_name="customers",  
    index="customer_id",             
)

es.normalize_dataframe(
    base_dataframe_name="data", 
    new_dataframe_name="items",  
    index="stock_code",          
)

es["data"].shape
es["data"].head()

es["invoices"].shape
es["invoices"].head()

es["customers"].shape
es["customers"].head()

es["items"].shape
es["items"].head()

feature_matrix, feature_defs = ft.dfs(
    entityset=es,                        
    target_dataframe_name="customers",   
    ignore_columns={                     
        "invoices":["invoice"],
        "data":["customer_id"],
    }
)

print(len(feature_defs))

feature_defs[5:10]
feature_matrix.head()

feature_matrix[feature_matrix.columns[5:10]].head()

feature_matrix, feature_defs = ft.dfs(
    entityset=es,                     
    target_dataframe_name="invoices",  
    ignore_columns = {                
        "data": ["customer_id"],
    }, 
    max_depth = 1,
)

print(len(feature_defs))

feature_matrix.head()
feature_matrix, feature_defs = ft.dfs(
    entityset=es,                   
    target_dataframe_name="items",   
    ignore_columns = {               
        "data": ["customer_id"]
    }, 
    verbose=True,
    max_depth = 1,
)

print(len(feature_defs))

feature_matrix.head()

