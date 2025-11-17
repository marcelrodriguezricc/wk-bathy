from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class AOI:
    name: str
    filename: str
    lat: str
    lon: str
    crm_link: str
    crm_local: str = "null"
    crm_date: str = "null"
    bbox_lat: float = 0.25
    bbox_lon: float = 0.25
    date_offset: int = 0

    swh_array: Optional[np.ndarray] = None
    swh_dates: Optional[list] = None
    optical_dates: Optional[list] = None

@dataclass
class params:
    num_days: int = 20