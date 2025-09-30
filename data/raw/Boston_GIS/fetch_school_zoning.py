import pandas as pd
from shapely import wkt
from shapely.geometry import Point


output_file = "boston_schools_and_offices.csv"
public_schools_url = "https://gisportal.boston.gov/arcgis/rest/services/Education/OpenData/MapServer/0/query"
nonpublic_schools_url = "https://gisportal.boston.gov/arcgis/rest/services/Education/OpenData/MapServer/1/query"
zoning_csv_url = "https://data.boston.gov/dataset/61396e0c-f1e6-4625-ab38-08e9e7e355bd/resource/0833f3e5-99a3-49ef-8798-adffdc631ce8/download/boston_zoning_subdistricts.csv"
def fetch_arcgis_features(url, entity_type):
    import requests
    from shapely.geometry import shape

    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "geojson",
        "resultRecordCount": 10000
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    rows = []
    for feature in data["features"]:
        props = feature["properties"]
        geom = shape(feature["geometry"])

        if geom.geom_type == "Point":
            lon, lat = geom.x, geom.y
        else:
            lon, lat = geom.centroid.x, geom.centroid.y
        
        name = props.get("SCH_NAME") or props.get("SCHOOL_NAME") or props.get("NAME") or "Unknown"
        rows.append({
            "name": name,
            "type": entity_type,
            "latitude": lat,
            "longitude": lon
        })
    return rows

def fetch_offices_from_zoning(csv_url):
    """Fetch office/commercial/mixed-use zoning areas from CSV"""
    df = pd.read_csv(csv_url)
    rows = []

    for _, row in df.iterrows():
        zoning_code = str(row.get("ZONING") or "")
        geom_wkt = row.get("geometry")  # WKT polygon/point

        if any(code in zoning_code for code in ["COM", "OFF", "MU"]):
            if pd.notna(geom_wkt):
                geom = wkt.loads(geom_wkt)
                if geom.geom_type in ["Polygon", "MultiPolygon"]:
                    lon, lat = geom.centroid.x, geom.centroid.y
                elif geom.geom_type == "Point":
                    lon, lat = geom.x, geom.y
                else:
                    continue
                rows.append({
                    "name": f"Zoning_{zoning_code}",
                    "type": "office",
                    "latitude": lat,
                    "longitude": lon
                })
    return rows

public_schools = fetch_arcgis_features(public_schools_url, "public_school")
print(f"Found {len(public_schools)} public schools.")

nonpublic_schools = fetch_arcgis_features(nonpublic_schools_url, "nonpublic_school")
print(f"Found {len(nonpublic_schools)} non-public schools.")

offices = fetch_offices_from_zoning(zoning_csv_url)
print(f"Found {len(offices)} office/commercial areas.")
all_entities = public_schools + nonpublic_schools + offices
df_entities = pd.DataFrame(all_entities)
df_entities.to_csv(output_file, index=False)
print(f"Saved combined schools and offices data to {output_file}")
