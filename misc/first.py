import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import googlemaps
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class GoogleMapsTransportOptimizer:
    def __init__(self, api_key, office_location):
        """
        Initialize the transport optimizer with Google Maps client
        """
        self.gmaps = googlemaps.Client(key=api_key)
        self.office_location = office_location
    
    def load_data(self, uploaded_file):
        """
        Load and validate employee location data
        """
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Required columns validation
            required_columns = ['staff_id', 'name', 'latitude', 'longitude', 'address']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing columns: {', '.join(missing_columns)}")
            
            # Coordinate validation
            invalid_coords = df[
                (df['latitude'].abs() > 90) | 
                (df['longitude'].abs() > 180)
            ]
            
            if not invalid_coords.empty:
                raise ValueError("Invalid latitude or longitude values detected")
            
            return df
        
        except Exception as e:
            st.error(f"Data loading error: {e}")
            return None
    
    def get_google_route_details(self, origin, destination):
        """
        Get detailed route information using Google Maps
        """
        try:
            # Request directions
            directions = self.gmaps.directions(
                origin, 
                destination, 
                mode='driving',
                traffic_model='best_guess',
                departure_time='now'
            )
            
            if directions:
                route = directions[0]['legs'][0]
                return {
                    'distance': route['distance']['text'],
                    'duration': route['duration']['text'],
                    'duration_in_traffic': route['duration_in_traffic']['text'] if 'duration_in_traffic' in route else 'N/A'
                }
            return None
        
        except Exception as e:
            st.warning(f"Route calculation error: {e}")
            return None
    
    def optimize_routes(self, df, max_passengers_per_car):
        """
        Optimize routes using advanced clustering techniques
        """
        # Prepare coordinates for clustering
        X = df[['latitude', 'longitude']].values
        
        # Normalize coordinates
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters
        n_clusters = max(1, int(np.ceil(len(df) / max_passengers_per_car)))
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Organize routes
        routes = []
        for cluster in range(n_clusters):
            cluster_df = df[df['cluster'] == cluster]
            
            # Calculate cluster centroid
            centroid_lat = cluster_df['latitude'].mean()
            centroid_lon = cluster_df['longitude'].mean()
            centroid_location = f"{centroid_lat},{centroid_lon}"
            
            # Calculate routes and total distance
            cluster_routes = []
            total_distance = 0
            
            for _, employee in cluster_df.iterrows():
                employee_location = f"{employee['latitude']},{employee['longitude']}"
                
                # Get route details
                route_details = self.get_google_route_details(
                    employee_location, 
                    f"{self.office_location[0]},{self.office_location[1]}"
                )
                
                if route_details:
                    cluster_routes.append({
                        'employee': employee,
                        'route_details': route_details
                    })
                    # Parse distance (assumes format like "10.5 km")
                    total_distance += float(route_details['distance'].split()[0])
            
            routes.append({
                'car_number': cluster + 1,
                'employees': cluster_df,
                'centroid': (centroid_lat, centroid_lon),
                'total_distance': total_distance,
                'detailed_routes': cluster_routes
            })
        
        return routes
    
    def visualize_routes(self, df, routes):
        """
        Create interactive route visualization
        """
        # Create base map
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        route_map = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Color palette for routes
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkgreen']
        
        # Add office location marker
        folium.Marker(
            self.office_location,
            popup='Office Location',
            icon=folium.Icon(color='black', icon='briefcase')
        ).add_to(route_map)
        
        # Visualize routes
        for route in routes:
            color = colors[(route['car_number'] - 1) % len(colors)]
            
            # Add centroid marker
            folium.Marker(
                route['centroid'],
                popup=f"Car {route['car_number']} Centroid",
                icon=folium.Icon(color=color, icon='car')
            ).add_to(route_map)
            
            # Add employee markers and routes
            for route_info in route['detailed_routes']:
                employee = route_info['employee']
                route_details = route_info['route_details']
                
                # Employee marker
                folium.Marker(
                    [employee['latitude'], employee['longitude']],
                    popup=f"{employee['name']} - Car {route['car_number']}<br>{route_details['distance']}<br>{route_details['duration']}",
                    icon=folium.Icon(color=color)
                ).add_to(route_map)
        
        return route_map

def main():
    st.set_page_config(page_title="Staff Transport Optimizer", page_icon="🚐")
    
    # Title and description
    st.title("🚐 Google Maps Staff Transportation Route Optimizer")
    st.markdown("""
    Advanced route planning with Google Maps:
    - Real-time route and traffic information
    - Optimize vehicle routes
    - Minimize total travel distance
    """)
    
    # Sidebar inputs
    st.sidebar.header("Optimization Parameters")
    
    # API Key input (with default from environment)
    api_key = st.sidebar.text_input(
        "Google Maps API Key", 
        value='AIzaSyC8Vzow3LdOOKZByIetQ4LV-vQEuSQk9Mc',
        type="password"
    )
    
    # Office location input (with default from environment)
    office_lat = st.sidebar.number_input(
        "Office Latitude", 
        value=5.582636441579255,
        format="%.10f"
    )
    office_lon = st.sidebar.number_input(
        "Office Longitude", 
        value=-0.143551646497661,
        format="%.10f"
    )
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Staff Location CSV", 
        type=['csv'],
        help="CSV should contain: staff_id, name, latitude, longitude, address"
    )
    
    # Max passengers input
    max_passengers = st.sidebar.slider(
        "Maximum Passengers per Car", 
        min_value=2, 
        max_value=6, 
        value=4, 
        help="Maximum number of staff per vehicle"
    )
    
    # Optimization process
    if uploaded_file is not None:
        try:
            # Initialize optimizer
            optimizer = GoogleMapsTransportOptimizer(
                api_key, 
                (office_lat, office_lon)
            )
            
            # Load data
            df = optimizer.load_data(uploaded_file)
            
            if df is not None:
                # Optimize routes
                routes = optimizer.optimize_routes(df, max_passengers)
                
                # Visualization
                route_map = optimizer.visualize_routes(df, routes)
                
                # Display results
                st.subheader("🗺️ Route Optimization Results")
                
                # Detailed route information
                for route in routes:
                    st.markdown(f"### 🚦 Car {route['car_number']}")
                    
                    # Prepare route details
                    route_details = []
                    for route_info in route['detailed_routes']:
                        employee = route_info['employee']
                        details = route_info['route_details']
                        route_details.append({
                            'Name': employee['name'],
                            'Address': employee['address'],
                            'Distance': details['distance'],
                            'Duration': details['duration'],
                            'Duration in Traffic': details['duration_in_traffic']
                        })
                    
                    # Display route details
                    route_df = pd.DataFrame(route_details)
                    st.dataframe(route_df)
                    st.write(f"**Total Route Distance:** {route['total_distance']:.2f} km")
                
                # Interactive map
                st.subheader("📍 Route Visualization")
                st_folium(route_map, width=700, height=500)
        
        except Exception as e:
            st.error(f"Optimization error: {e}")

if __name__ == "__main__":
    main()