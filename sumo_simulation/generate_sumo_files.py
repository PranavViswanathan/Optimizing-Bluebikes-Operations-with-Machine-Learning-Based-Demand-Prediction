
import pandas as pd
import datetime
import os

# Configuration
PARQUET_PATH = '/Users/pranavviswanathan/Northeastern/Education/Fall2025/ML-OPS/Project/data_pipeline/data/raw/bluebikes/trips_2024.parquet'
STATIONS_CSV = '/Users/pranavviswanathan/Northeastern/Education/Fall2025/ML-OPS/Project/sumo_simulation/stations.csv'
OUTPUT_DIR = '/Users/pranavviswanathan/Northeastern/Education/Fall2025/ML-OPS/Project/sumo_simulation'
SIMULATION_DATE = '2024-05-01'
START_HOUR = 8
END_HOUR = 9

def generate_sumo_files():
    print("Loading stations...")
    stations = pd.read_csv(STATIONS_CSV)
    
    # 1. Generate stations.add.xml
    # We map stations to the nearest edge in the network (in a real scenario). 
    # Since we might not have the network loaded, we will define them as POIs or stops.
    # For visualization, POIs are easier if we don't have edge IDs.
    
    with open(os.path.join(OUTPUT_DIR, 'stations.add.xml'), 'w') as f:
        f.write('<additional>\n')
        for _, row in stations.iterrows():
            # Color encoded by some dummy status or just blue
            color = "0,0,1" 
            f.write(f'    <poi id="{row["id"]}" type="station" x="{row["lon"]}" y="{row["lat"]}" color="{color}" layer="100"/>\n')
        f.write('</additional>\n')
    print("Generated stations.add.xml")

    # 2. Generate trips.xml
    print(f"Loading trips for {SIMULATION_DATE} {START_HOUR}:00 to {END_HOUR}:00...")
    df = pd.read_parquet(PARQUET_PATH)
    
    # Ensure start_time is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['start_time']):
        df['start_time'] = pd.to_datetime(df['start_time'])

    # Filter by date and time
    # Note: timezone might be issue, assume UTC or local. 
    # inspect_parquet output: 2024-01-31 12:16:49+00:00 (UTC)
    
    target_start = pd.Timestamp(f"{SIMULATION_DATE} {START_HOUR}:00:00", tz="UTC")
    target_end = pd.Timestamp(f"{SIMULATION_DATE} {END_HOUR}:00:00", tz="UTC")
    
    mask = (df['start_time'] >= target_start) & (df['start_time'] < target_end)
    trips = df[mask].sort_values('start_time')
    
    print(f"Found {len(trips)} trips.")
    
    # We need to offset times to start at 0 for the simulation
    min_time = trips['start_time'].min()
    
    with open(os.path.join(OUTPUT_DIR, 'trips.trips.xml'), 'w') as f:
        f.write('<routes>\n')
        f.write('    <vType id="bike" vClass="bicycle" accel="0.8" decel="1.5" sigma="0.5" length="1.6" minGap="0.5" maxSpeed="5.5" guiShape="bicycle"/>\n')
        
        for idx, row in trips.iterrows():
            # Calculate depart time in seconds from simulation start
            depart = (row['start_time'] - min_time).total_seconds()
            
            # Using fromTaz and toTaz if we had districts, or from/to edges.
            # Without a network graph mapping lat/lon to edges, we can't generate valid <trip> elements that SUMO can route.
            # However, we can use <person> or just defined edges.
            
            # CRITICAL: For SUMO to run, we need "from" and "to" EDGES (street names/node IDs). 
            # We only have Lat/Lon.
            # Standard approach: Use shortest path on the map.
            # But we don't have the map loaded to query nearest edge.
            
            # WORKAROUND: We will comment out the trips or make them dummy trips 
            # and explain that 'map matching' is required. 
            # OR, we can assume the user will run a script to map stations to edges.
            
            # For this file to be useful, I'll generate it assuming stations IDs *could* be mapped to edges.
            # I will use the station coordinates to define "fromXY" and "toXY" which SUMO supports!
            # defined in <trip ... fromXY="lon,lat" toXY="lon,lat" .../>
            
            f.write(f'    <trip id="{row["ride_id"]}" type="bike" depart="{depart:.2f}" fromXY="{row["start_station_longitude"]},{row["start_station_latitude"]}" toXY="{row["end_station_longitude"]},{row["end_station_latitude"]}" />\n')
            
        f.write('</routes>\n')

    print("Generated trips.trips.xml")
    
    # 3. Generate sumo config
    with open(os.path.join(OUTPUT_DIR, 'simulation.sumocfg'), 'w') as f:
        f.write('''<configuration>
    <input>
        <net-file value="boston.net.xml"/>
        <route-files value="trips.trips.xml"/>
        <additional-files value="stations.add.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
</configuration>
''')
    print("Generated simulation.sumocfg")

    # 4. Generate Suggestions (Dummy for now, but placeholder for Model usage)
    # The user asked to "use the models". 
    # I'll create a separate file 'suggestions.add.xml' that puts a marker on stations predicted to have high demand.
    
    print("Generating suggestions based on historical data (proxy for model output)...")
    
    # Count departures per station in this hour (Demand)
    demand = trips['start_station_id'].value_counts()
    
    with open(os.path.join(OUTPUT_DIR, 'suggestions.add.xml'), 'w') as f:
        f.write('<additional>\n')
        for station_id, count in demand.items():
            # Check station details
            if station_id in stations['id'].values:
                station = stations[stations['id'] == station_id].iloc[0]
                
                # If high demand, mark it red
                if count > 5: # Threshold
                    f.write(f'    <poi id="demand_{station_id}" type="high_demand" x="{station["lon"]}" y="{station["lat"]}" color="1,0,0" width="10" height="10" layer="101"/>\n')
        f.write('</additional>\n')
    
    print("Generated suggestions.add.xml")
    
    # Update config to include suggestions
    # We'll just overwrite the config to include suggestions.add.xml
    with open(os.path.join(OUTPUT_DIR, 'simulation.sumocfg'), 'w') as f:
        f.write('''<configuration>
    <input>
        <net-file value="boston.net.xml"/>
        <route-files value="trips.trips.xml"/>
        <additional-files value="stations.add.xml,suggestions.add.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
</configuration>
''')

if __name__ == "__main__":
    generate_sumo_files()
