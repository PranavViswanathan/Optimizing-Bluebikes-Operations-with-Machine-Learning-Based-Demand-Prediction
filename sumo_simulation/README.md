# SUMO Simulation for Boston Bluebikes

This folder contains a SUMO (Simulation of Urban MObility) simulation for Boston's Bluebikes system.

## Files Generated

- **`stations.csv`**: List of all 530 unique Bluebikes stations with their IDs, names, and coordinates (lat/lon)
- **`stations.add.xml`**: SUMO additional file defining station locations as POIs (Points of Interest) on the map
- **`trips.trips.xml`**: SUMO trips file containing 1,264 actual bike trips from May 1, 2024, 8:00-9:00 AM
- **`suggestions.add.xml`**: High-demand station markers (red POIs) based on historical trip data - stations with >5 departures are highlighted
- **`simulation.sumocfg`**: SUMO configuration file that ties everything together

## How to Run the Simulation

### Prerequisites

1. **Install SUMO**: Download and install SUMO from [https://sumo.dlr.de/docs/Installing/index.html](https://sumo.dlr.de/docs/Installing/index.html)

2. **Get the Boston Road Network**:
   
   You need a SUMO network file (`boston.net.xml`) for Boston. You can generate this using OpenStreetMap data:

   ```bash
   # Install osmWebWizard dependencies (if not already installed)
   pip install sumolib traci
   
   # Download Boston area from OpenStreetMap and convert to SUMO network
   # Option 1: Use osmWebWizard.py (GUI tool included with SUMO)
   osmWebWizard.py
   # Then select Boston area on the map and export the network
   
   # Option 2: Use netconvert directly with OSM data
   # First, download Boston OSM data from https://www.openstreetmap.org/export
   # or use: wget https://download.bbbike.org/osm/bbbike/Boston/Boston.osm.gz
   # Then convert:
   netconvert --osm-files Boston.osm.gz -o boston.net.xml
   ```

   Place the resulting `boston.net.xml` file in this `sumo_simulation` folder.

### Running the Simulation

Once you have `boston.net.xml`:

```bash
# Run with GUI (visual)
sumo-gui -c simulation.sumocfg

# Run headless (faster, no visualization)
sumo -c simulation.sumocfg
```

## What the Simulation Shows

- **Blue POIs**: Bluebikes station locations (530 stations across Boston)
- **Red POIs**: High-demand stations (predicted based on historical data showing >5 departures in the hour)
- **Bike trips**: 1,264 actual bike trips from the morning rush hour (8-9 AM on May 1, 2024)

The trips use `fromXY` and `toXY` coordinates, which SUMO will map to the nearest edges in the road network.

## Integration with ML Models

The `suggestions.add.xml` file currently uses a simple threshold (>5 trips) to identify high-demand stations. 

To integrate with your trained XGBoost model:

1. The model expects 48 features (see `inspect_model.py` output)
2. You can modify `generate_sumo_files.py` to:
   - Load the model from `../model_pipeline/models/production/current_model.pkl`
   - Generate predictions for each station at each time step
   - Color-code stations based on predicted demand levels
   - Generate rebalancing suggestions (e.g., arrows showing where bikes should be moved)

## Scripts in This Folder

- **`extract_stations.py`**: Extracts unique stations from the 2024 trip data
- **`generate_sumo_files.py`**: Generates all SUMO XML files for the simulation
- **`inspect_model.py`**: Shows the features expected by the trained XGBoost model
- **`inspect_parquet.py`**: Inspects the structure of the trip data
- **`check_date.py`**: Checks the date range available in the dataset

## Next Steps

1. Install SUMO
2. Generate or download the Boston road network (`boston.net.xml`)
3. Run the simulation with `sumo-gui -c simulation.sumocfg`
4. Optionally: Enhance `generate_sumo_files.py` to use ML model predictions for demand forecasting
