from pathlib import Path
from typing import Union
import pandas as pd

def log_file_status(log_path: Union[str, Path], filename: str, status: bool):
    """Append filename and read status to a CSV log."""
    log_path = Path(log_path)
    entry = pd.DataFrame([{"filename": filename, "read": status}])
    if not log_path.exists():
        entry.to_csv(log_path, index=False)
    else:
        entry.to_csv(log_path, mode="a", header=False, index=False)
