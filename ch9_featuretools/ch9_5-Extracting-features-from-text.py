import pandas as pd
import featuretools as ft
from woodwork.logical_types import Categorical, NaturalLanguage

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
        "description": NaturalLanguage, 
    },
)

es.normalize_dataframe(
    base_dataframe_name="data",    
    new_dataframe_name="invoices", 
    index="invoice",              
    copy_columns=["customer_id"],  
)

text_primitives = ["num_words", "num_characters", "MeanCharactersPerWord" , "PunctuationCount"]

feature_matrix, feature_defs = ft.dfs(
    entityset=es,                      
    target_dataframe_name="data",      
    agg_primitives=[],                
    trans_primitives=text_primitives, 
    ignore_dataframes=["invoices"],
)

text_f = [
    "NUM_CHARACTERS(description)",
    "NUM_WORDS(description)",
    "PUNCTUATION_COUNT(description)",
]

feature_matrix[text_f].head()

from nlp_primitives import (
    DiversityScore,
    MeanCharactersPerSentence,
    PartOfSpeechCount,
)
text_primitives = [
    DiversityScore,
#     MeanCharactersPerSentence,
#     PartOfSpeechCount,
]

feature_matrix, feature_defs = ft.dfs(
    entityset=es,                     
    target_dataframe_name="data",     
    agg_primitives=[],               
    trans_primitives=text_primitives,  
    ignore_dataframes=["invoices"],
)

feature_matrix.head()

