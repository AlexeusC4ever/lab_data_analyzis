import pandas as pd  
        
def get_indexes_of_cat_columns(df: pd.DataFrame, col_names: str):
    idx = []
    for col in col_names:
        idx.append(df.columns.tolist().index(col))
        
    return idx