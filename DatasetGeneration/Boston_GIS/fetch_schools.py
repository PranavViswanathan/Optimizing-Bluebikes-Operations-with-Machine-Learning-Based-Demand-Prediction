import requests
import pandas as pd
from shapely.geometry import shape

# -----------------------------
# CONFIG
# -----------------------------
output_file = "boston_schools.csv"

# ArcGIS REST endpoints for schools
public_schools_url = "https://gisportal.boston.gov/arcgis/rest/services/Education/OpenData/MapServer/0/query"
nonpublic_schools_url = "https://gisportal.boston.gov/arcgis/rest/services/Education/OpenData/MapServer/1/query"

def fetch_arcgis_features(url, entity_type):
    """Fetch all features from an ArcGIS REST service layer"""
    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "geojson",
        "resultRecordCount": 10000  # fetch up to 10k records
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    rows = []
    for feature in data["features"]:
        props = feature["properties"]
        geom = shape(feature["geometry"])  # shapely geometry

        if geom.geom_type == "Point":
            lon, lat = geom.x, geom.y
        elif geom.geom_type in ["Polygon", "MultiPolygon"]:
            lon, lat = geom.centroid.x, geom.centroid.y
        else:
            continue
        
        name = props.get("SCH_NAME") or props.get("SCHOOL_NAME") or props.get("NAME") or "Unknown"
        rows.append({
            "name": name,
            "type": entity_type,
            "latitude": lat,
            "longitude": lon
        })
    return rows

# Fetch public and non-public schools
public_schools = fetch_arcgis_features(public_schools_url, "public_school")
print(f"Found {len(public_schools)} public schools.")

nonpublic_schools = fetch_arcgis_features(nonpublic_schools_url, "nonpublic_school")
print(f"Found {len(nonpublic_schools)} non-public schools.")

# Combine and save CSV
all_schools = public_schools + nonpublic_schools
df_schools = pd.DataFrame(all_schools)
df_schools.to_csv(output_file, index=False)
print(f"Saved schools data to {output_file}")
