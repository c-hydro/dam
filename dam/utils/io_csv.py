import pandas as pd

def read_csv(src: str) -> pd.DataFrame:
    """
    Read a csv as DataFrame.
    """
    destination = pd.read_csv(src)

    return destination


