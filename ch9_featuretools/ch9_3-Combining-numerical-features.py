import pandas as pd
import featuretools as ft
from woodwork.logical_types import Categorical

df = pd.read_csv("retail.csv", parse_dates=["invoice_date"])

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

ft.get_valid_primitives(es, target_dataframe_name="data", max_depth=2)

feature_matrix, feature_defs = ft.dfs(
    entityset=es,                          
    target_dataframe_name="data",         
    agg_primitives=[],                    
    trans_primitives=["multiply_numeric"], 
    primitive_options={                   
        ("multiply_numeric"): {
            'include_columns': {
                'data': ["quantity", "price"]
            }
        }
    },
    ignore_dataframes=["invoices"],
)

feature_matrix.head()

df = pd.read_csv("retail.csv", parse_dates=["invoice_date"])
df.head()

df["amount"] = df["quantity"].mul(df["price"])
df.head()

