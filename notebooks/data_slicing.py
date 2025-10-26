import pandas as pd
import matplotlib.pyplot as plt
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_bias_audit(file_path, dataset_type="bluebikes"):
    """
    dataset_type: "bluebikes" or "college"
    """
    save_path = os.path.join(PROJECT_DIR,'data_pipeline', 'assets')
    
    df = pd.read_pickle(file_path)
    print(f"\n=== {dataset_type.upper()} DATA BIAS AUDIT REPORT ===")

    # ----------------------- BLUEBIKES -----------------------
    if dataset_type == "bluebikes":
        # Convert time column
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')

        # Representation Bias
        print("\n--- 1. Representation Bias (User Type) ---")
        if 'user_type' in df.columns:
            print(df['user_type'].value_counts(normalize=True) * 100)
            df['user_type'].value_counts(normalize=True).plot(kind='bar', title="User Type Distribution")
            plt.ylabel("Percentage")
            plt.savefig(os.path.join(save_path,'bluebikes_user_type_distribution.png'), dpi=100, bbox_inches='tight')
            plt.close()
        else:
            print("No 'user_type' column found. Skipping this check.")

        # Temporal Bias
        print("\n--- 2. Temporal Bias (Month & Hour) ---")
        df['month'] = df['start_time'].dt.month
        df['hour'] = df['start_time'].dt.hour
        print("\nTrips by Month (%):")
        print(df['month'].value_counts(normalize=True).sort_index() * 100)
        df['month'].value_counts().sort_index().plot(kind='bar', title="Trips per Month")
        plt.ylabel("Percentage")
        plt.savefig(os.path.join(save_path,'bluebikes_trips_by_month.png'), dpi=100, bbox_inches='tight')
        plt.close()
        print("\nTrips by Hour (%):")
        print(df['hour'].value_counts(normalize=True).sort_index() * 100)
        df['hour'].value_counts().sort_index().plot(kind='line', title="Trips by Hour")
        plt.ylabel("Count")
        plt.savefig(os.path.join(save_path,'bluebikes_trips_by_hour.png'), dpi=100, bbox_inches='tight')
        plt.close()

        # Geographic Bias
        print("\n--- 3. Geographic Bias (Top Stations) ---")
        if 'start_station_name' in df.columns:
            print(df['start_station_name'].value_counts().head(10))
            df['start_station_name'].value_counts().head(10).plot(kind='bar', title="Top 10 Start Stations")
            plt.ylabel("Trips")
            plt.savefig(os.path.join(save_path,'bluebikes_top_stations.png'), dpi=100, bbox_inches='tight')
            plt.close()
        else:
            print("No 'start_station_name' column found. Skipping this check.")

        # Outliers
        if 'duration' in df.columns:
            print("\n--- 6. Outlier Bias (Trip Duration) ---")
            print(df['duration'].describe(percentiles=[0.01, 0.99]))
            outliers = df[df['duration'] > df['duration'].quantile(0.99)]
            print("Outliers above 99th percentile:", len(outliers))
            negative_or_zero = df[df['duration'] <= 0]
            print("Negative or zero durations:", len(negative_or_zero))
        else:
            print("No 'duration' column found. Skipping outlier check.")

     # ----------------------- COLLEGE DATA -----------------------
    elif dataset_type == "college":
        print("\n=== COLLEGE DATA BIAS AUDIT (FOR BIKESHARE DEMAND PREDICTION) ===")
        
        # Set up figure for multiple plots
        fig = plt.figure(figsize=(16, 12))
        
        # ----------------------- 1. Geographic Distribution -----------------------
        print("\n--- 1. Geographic Distribution (Critical for proximity analysis) ---")

        # City/Neighborhood distribution
        if 'City' in df.columns:
            city_counts = df['City'].value_counts()
            print(f"\nTop neighborhoods by college count:")
            print(city_counts.head(10))
            
            # Plot 2: Neighborhood distribution (horizontal bar chart)
            ax2 = fig.add_subplot(2, 2, 1)
            top_10_cities = city_counts.head(10)
            colors = ['red' if x > len(df) * 0.2 else 'steelblue' for x in top_10_cities.values]
            top_10_cities.plot(kind='barh', ax=ax2, color=colors)
            ax2.set_title("College Distribution by Neighborhood\n(Red = >20% concentration)")
            ax2.set_xlabel("Number of Colleges")
            ax2.set_ylabel("")
            
            # Check concentration in single neighborhood
            top_neighborhood_pct = city_counts.iloc[0] / len(df) * 100
            if top_neighborhood_pct > 30:
                print(f"\n Geographic concentration: {top_neighborhood_pct:.1f}% of colleges in {city_counts.index[0]}")
                print("   This may bias proximity-based predictions toward this area")
            
            # Note about empty City values
            empty_city = df['City'].isna().sum() + (df['City'] == '').sum()
            if empty_city > 0:
                print(f" {empty_city} colleges have empty City field")

        # ----------------------- 2. Student Population (Demand Driver) -----------------------
        print("\n--- 2. Student Population Distribution (Key demand driver) ---")
        
        if 'NumStudent' in df.columns:
            # Filter out zeros for meaningful statistics
            students_nonzero = df[df['NumStudent'] > 0]['NumStudent']
            
            print("\nStudent count statistics (excluding zeros):")
            print(students_nonzero.describe())
            
            # Check for zeros and missing
            zero_students = (df['NumStudent'] == 0).sum()
            missing_students = df['NumStudent'].isnull().sum()
            
            if zero_students > 0:
                print(f"\n⚠️ {zero_students} colleges have 0 students - likely data quality issue")
            if missing_students > 0:
                print(f"⚠️ {missing_students} colleges missing student count")
            
            # Plot 3: Student population distribution (histogram with log scale option)
            ax3 = fig.add_subplot(2, 2, 2)
            if len(students_nonzero) > 0:
                ax3.hist(students_nonzero, bins=30, color='darkgreen', alpha=0.7, edgecolor='black')
                ax3.axvline(students_nonzero.median(), color='red', linestyle='--', 
                           label=f'Median: {students_nonzero.median():.0f}')
                ax3.set_title("Student Population Distribution\n(Excluding zeros)")
                ax3.set_xlabel("Number of Students")
                ax3.set_ylabel("Number of Colleges")
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            
            # # Identify size categories
            if len(students_nonzero) > 0:
                df['size_category'] = pd.cut(df['NumStudent'], 
                                            bins=[-1, 0, 1000, 5000, 10000, 50000],
                                            labels=['No Data/Zero', 'Small(<1k)', 'Medium(1-5k)', 'Large(5-10k)', 'Very Large(>10k)'])
                
                # Plot 5: Size category pie chart
                # ax5 = fig.add_subplot(3, 3, 5)
                size_dist = df['size_category'].value_counts()
                # colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
                # wedges, texts, autotexts = ax5.pie(size_dist.values, 
                #                                     labels=size_dist.index, 
                #                                     colors=colors_pie,
                #                                     autopct='%1.1f%%',
                #                                     startangle=90)
                # ax5.set_title("College Size Distribution")
                
                print("\nCollege size distribution:")
                for cat in size_dist.index:
                    print(f"  {cat}: {size_dist[cat]} ({size_dist[cat]/len(df)*100:.1f}%)")
        

        # ----------------------- 4. Geographic-Student Distribution -----------------------
        print("\n--- 4. Problematic Data Slices for Demand Prediction ---")
        
        # Analyze geographic concentration with student population
        if 'City' in df.columns and 'NumStudent' in df.columns:
            # Plot 7: Student population by neighborhood
            ax7 = fig.add_subplot(2, 2, 3)
            neighborhood_students = df.groupby('City')['NumStudent'].sum().sort_values(ascending=False).head(10)
            
            neighborhood_students.plot(kind='bar', ax=ax7, color='purple', alpha=0.7)
            ax7.set_title("Total Student Population by Neighborhood")
            ax7.set_xlabel("")
            ax7.set_ylabel("Total Students")
            ax7.tick_params(axis='x', rotation=45)
            ax7.grid(True, alpha=0.3, axis='y')
            
            print("\nTop neighborhoods by total student population:")
            print(neighborhood_students.head(5))
            
            # Check if student population is concentrated
            total_students = df['NumStudent'].sum()
            if total_students > 0:
                top_neighborhood_students = neighborhood_students.iloc[0] if len(neighborhood_students) > 0 else 0
                concentration = top_neighborhood_students / total_students * 100
                if concentration > 40:
                    print(f"\n {concentration:.1f}% of all students are in {neighborhood_students.index[0]}")
                    print("   Proximity to this area will dominate demand predictions")
            
            # Plot 8: Heatmap of neighborhood vs size category
            if 'size_category' in df.columns:
                ax8 = fig.add_subplot(2, 2, 4)
                # Create pivot table for heatmap
                heatmap_data = df.pivot_table(index='City', columns='size_category', 
                                              aggfunc='size', fill_value=0)
                # Select top neighborhoods for visibility
                top_neighborhoods = df['City'].value_counts().head(8).index
                heatmap_data = heatmap_data.loc[heatmap_data.index.isin(top_neighborhoods)]
                
                im = ax8.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
                ax8.set_xticks(range(len(heatmap_data.columns)))
                ax8.set_yticks(range(len(heatmap_data.index)))
                ax8.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
                ax8.set_yticklabels(heatmap_data.index)
                ax8.set_title("College Size Distribution by Neighborhood")
                
                # Add colorbar
                plt.colorbar(im, ax=ax8, label='Count')
                
                # Add text annotations
                for i in range(len(heatmap_data.index)):
                    for j in range(len(heatmap_data.columns)):
                        text = ax8.text(j, i, str(heatmap_data.values[i, j]),
                                       ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path,'college_analysis_combined.png'), dpi=100, bbox_inches='tight')
        plt.close()


    # ----------------------- MISSINGNESS -----------------------
    print("\n--- 5. Missing Data Bias ---")
    print("Missing Values (%):")
    print(df.isna().mean().sort_values(ascending=False) * 100)

    print("\n=== END OF REPORT ===")
    print("\nGraphs have been saved to the 'graphs' folder.")



# ----------------------- RUN EXAMPLES -----------------------
if __name__ == "__main__":
    # BlueBikes dataset
    run_bias_audit("../data_pipeline/data/processed/bluebikes/raw_data.pkl", dataset_type="bluebikes")
    
    # College dataset
    run_bias_audit("../data_pipeline/data/processed/boston_clg/raw_data.pkl", dataset_type="college")