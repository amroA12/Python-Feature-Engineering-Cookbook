import pandas as pd
import featuretools as ft
from woodwork.logical_types import Categorical, NaturalLanguage

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
        "description": NaturalLanguage, 
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

agg_primitives = ["mean", "max", "min", "sum"]

feature_matrix, feature_defs = ft.dfs(
    entityset=es,                                 
    target_dataframe_name="customers",          
    agg_primitives=agg_primitives,               
    trans_primitives=[],                        
)

feature_matrix.head()

date_primitives = ["month", "weekday"]

text_primitives = ["num_words"]

trans_primitives = date_primitives + text_primitives 

agg_primitives = ["mean"]

feature_matrix, feature_defs = ft.dfs(
    entityset=es,                                
    target_dataframe_name="customers",           
    agg_primitives=agg_primitives,                
    trans_primitives=trans_primitives,            
    max_depth=3,
)

feature_matrix.head()

