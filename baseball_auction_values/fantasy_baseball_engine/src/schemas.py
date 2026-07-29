from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List
import pandas as pd

class HitterSchema(BaseModel):
    Name: str
    Pos: str
    PA: float
    AB: float
    H: float
    HR: float
    R: float
    RBI: float
    SB: float

class PitcherSchema(BaseModel):
    Name: str
    Pos: str
    IP: float
    W: float
    SV: float
    K: float
    ER: float
    BB: float
    HA: float

def validate_dataframe(df: pd.DataFrame, schema: BaseModel) -> pd.DataFrame:
    valid_rows = []
    for _, row in df.iterrows():
        try:
            valid_rows.append(schema(**row.to_dict()).dict())
        except ValidationError:
            # Skip invalid rows or fill with defaults in a real scenario
            continue
    return pd.DataFrame(valid_rows)
