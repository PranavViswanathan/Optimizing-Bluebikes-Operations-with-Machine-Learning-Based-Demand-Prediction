import pandas as pd

DEFAULT_MAPPING = {
    "tripduration": "duration",
    "starttime": "start_time",
    "stoptime": "stop_time",
    "start station id": "start_station_id",
    "start station name": "start_station_name",
    "start station latitude": "start_station_latitude",
    "start station longitude": "start_station_longitude",
    "end station id": "end_station_id",
    "end station name": "end_station_name",
    "end station latitude": "end_station_latitude",
    "end station longitude": "end_station_longitude",
    "bikeid": "bike_id",
    "usertype": "user_type",
    "birth year": "birth_year",
    "started_at": "start_time",
    "ended_at": "stop_time",
    "start_lat": "start_station_latitude",
    "start_lng": "start_station_longitude",
    "end_lat": "end_station_latitude",
    "end_lng": "end_station_longitude",
    "member_casual": "user_type",
}



def _normalize(col: str) -> str:
    """Lowercase, trim, and unify separators (spaces/underscores/hyphens)."""
    return (
        col.strip()
           .lower()
           .replace("-", " ")
           .replace("_", " ")
           .replace("  ", " ")
           .strip()
    )

def _normalized_mapping(raw_map: dict) -> dict:
    """Make mapping robust to case/spacing/underscore/hyphen drift."""
    return {_normalize(k): v for k, v in raw_map.items()}


def _rename_and_coalesce(df: pd.DataFrame, norm_map: dict) -> pd.DataFrame:
    """
    - normalize incoming column names
    - apply mapping to canonical names
    - coalesce duplicate targets (first non-null value)
    - tidy unmatched columns (normalized with spaces->underscores)
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    norm_cols = [_normalize(c) for c in df.columns]

    # Plan renames
    rename_plan = {}
    for orig, normed in zip(df.columns, norm_cols):
        if normed in norm_map:
            rename_plan[orig] = norm_map[normed]
        else:
            # keep normalized version (spaces -> underscores) for consistency
            rename_plan[orig] = normed.replace(" ", "_")

    df = df.rename(columns=rename_plan)

    # Coalesce duplicates after rename
    dup_targets = df.columns[df.columns.duplicated(keep=False)].unique()
    for tgt in dup_targets:
        same_cols = [c for c in df.columns if c == tgt]
        if len(same_cols) > 1:
            # take first non-null across duplicates
            df[tgt] = df[same_cols].bfill(axis=1).iloc[:, 0]
            # drop the extras (keep the first)
            for extra in same_cols[1:]:
                df.drop(columns=extra, inplace=True)

    return df

def _coerce_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # IDs can be alphanumeric → store as string
    for col in ["start_station_id", "end_station_id", "bike_id"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # Names / categories as string
    for col in ["start_station_name", "end_station_name", "user_type"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # Timestamps (handles both old/new schemas)
    for col in ["start_time", "stop_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Numerics (nullable-friendly)
    for col in ["duration", "birth_year",
                "start_station_latitude", "start_station_longitude",
                "end_station_latitude", "end_station_longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df