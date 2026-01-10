import pandas as pd
import featuretools as ft
from featuretools.primitives import IsFederalHoliday, DistanceToHoliday
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
    },
)

es.normalize_dataframe(
    base_dataframe_name="data",     
    new_dataframe_name="invoices", 
    index="invoice",                
    copy_columns=["customer_id"],  
)

es["invoices"].head()

is_bank_hol = IsFederalHoliday(country="UK")

hols = is_bank_hol.holidayUtil.federal_holidays.values()

available_hols = list(set(hols))

days_to_boxing = DistanceToHoliday(holiday="Boxing Day", country="UK")

date_primitives = [
    "day", "year", "month", "weekday",
    "days_in_month", "part_of_day",
    "hour", "minute",
    is_bank_hol,
    days_to_boxing
]

feature_matrix, feature_defs = ft.dfs(
    entityset=es,                      
    target_dataframe_name="invoices",  
    agg_primitives=[],                 
    trans_primitives=date_primitives,   
)

feature_matrix.head()

columns = [
    "DISTANCE_TO_HOLIDAY(first_data_time, holiday=Boxing Day, country=UK)",
    "HOUR(first_data_time)",
    "IS_FEDERAL_HOLIDAY(first_data_time, country=UK)",
]

feature_matrix[columns].head()

