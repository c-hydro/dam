import pandas as pd
import os

def read_csv(src: str) -> pd.DataFrame:
    """
    Read a csv as DataFrame.
    """
    destination = pd.read_csv(src)

    return destination

def save_csv(src: pd.DataFrame, destination: str) -> None:
    """
    Save a DataFrame to csv file.
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    src.to_csv(destination)


