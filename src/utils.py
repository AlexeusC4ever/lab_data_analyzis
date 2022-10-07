import pickle
import pandas as pd


def save_as_pickle(df: pd.DataFrame, path: str) -> None:
    df.to_pickle(path)

def load_as_pickle(path: str) -> pd.DataFrame:
    data_from_pickle = pd.read_pickle(path)
    return data_from_pickle
    
def save_model_as_pickle(model, path:str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)
        
def load_model_as_pickle(path:str):
    model = None
    
    with open(path, "rb") as f:
        model = pickle.load(f)
    
    return model
          
        