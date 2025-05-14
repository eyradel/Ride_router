import pandas as pd
import numpy as np
from geopy.distance import geodesic
from collections import defaultdict
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy.random as rnd
import googlemaps
import polyline
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class SOMCluster:
    def __init__(self, input_len, grid_size=3, sigma=1.0, learning_rate=0.5):
        self.grid_size = grid_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.input_len = input_len
        self._init_weights()
        
    def _init_weights(self):
        self.weights = rnd.rand(self.grid_size, self.grid_size, self.input_len)
        
    def _neighborhood(self, c, sigma):
        d = 2*sigma*sigma
        ax = np.arange(self.grid_size)
        xx, yy = np.meshgrid(ax, ax)
        return np.exp(-((xx-c[0])**2 + (yy-c[1])**2) / d)

    def find_winner(self, x):
        diff = self.weights - x
        dist = np.sum(diff**2, axis=-1)
        return np.unravel_index(np.argmin(dist), dist.shape)
    
    def train(self, data, epochs=2000):
        for epoch in range(epochs):
            sigma = self.sigma * (1 - epoch/epochs)
            lr = self.learning_rate * (1 - epoch/epochs)
            
            for x in data:
                winner = self.find_winner(x)
                g = self._neighborhood(winner, sigma)
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        self.weights[i,j] += lr * g[i,j] * (x - self.weights[i,j])

    def get_cluster(self, x):
        winner = self.find_winner(x)
        return winner[0] * self.grid_size + winner[1]

class StaffTransportOptimizer:
    def __init__(self, google_maps_key):
        self.office_location = {
            'lat': 5.582636441579255,
            'lon': -0.143551646497661
        }
        self.MAX_PASSENGERS = 4
        self.MIN_PASSENGERS = 3
        self.COST_PER_KM = 2.5
        self.scaler = MinMaxScaler()
        self.gmaps = googlemaps.Client(key=google_maps_key)

    def validate_staff_data(self, df):
        """Validate and clean staff location data"""
        try:
            # Check required columns
            required_columns = ['staff_id', 'name', 'latitude', 'longitude', 'address']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Create a copy to avoid modifying the original dataframe
            clean_df = df.copy()
            
            # Convert coordinates to numeric values
            clean_df['latitude'] = pd.to_numeric(clean_df['latitude'], errors='coerce')
            clean_df['longitude'] = pd.to_numeric(clean_df['longitude'], errors='coerce')
            
            # Remove rows with invalid coordinates
            clean_df = clean_df.dropna(subset=['latitude', 'longitude'])
            
            # Validate coordinate ranges for Ghana
            valid_lat_range = (4.5, 11.5)  # Ghana's latitude range
            valid_lon_range = (-3.5, 1.5)  # Ghana's longitude range
            
            # Create mask for valid coordinates
            coord_mask = (
                (clean_df['latitude'].between(*valid_lat_range)) &
                (clean_df['longitude'].between(*valid_lon_range))
            )
            
            clean_df = clean_df[coord_mask]
            
            # Remove duplicates based on staff_id
            clean_df = clean_df.drop_duplicates(subset=['staff_id'], keep='first')
            
            # Validate minimum required staff
            if len(clean_df) < self.MIN_PASSENGERS:
                raise ValueError(
                    f"Need at least {self.MIN_PASSENGERS} valid staff locations, "
                    f"but only found {len(clean_df)}"
                )
            
            # Validate data types
            clean_df['staff_id'] = clean_df['staff_id'].astype(str)
            clean_df['name'] = clean_df['name'].astype(str)
            clean_df['address'] = clean_df['address'].astype(str)
            
            # Add distance to office column
            clean_df['distance_to_office'] = clean_df.apply(
                lambda row: geodesic(
                    (row['latitude'], row['longitude']),
                    (self.office_location['lat'], self.office_location['lon'])
                ).km,
                axis=1
            )
            
            return clean_df
            
        except Exception as e:
            print(f"Error validating staff data: {str(e)}")
            return None

    def load_sample_data(self):
        """Load sample staff location data for testing"""
        try:
            # Try to load from the data directory
            data_path = os.path.join('data', 'Rider update.csv')
            if not os.path.exists(data_path):
                # If file doesn't exist, return a small sample dataset
                return pd.DataFrame({
                    'staff_id': ['1', '2', '3'],
                    'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
                    'latitude': [5.5826, 5.5926, 5.5726],
                    'longitude': [-0.1435, -0.1335, -0.1535],
                    'address': ['Accra', 'Accra', 'Accra']
                })
            
            samp = pd.read_csv(data_path)
            sample_data = pd.DataFrame({
                'staff_id': samp['staff_id'],
                'name': samp['name'],
                'latitude': samp['latitude'],
                'longitude': samp['longitude'],
                'address': samp['address']
            })
            return self.validate_staff_data(sample_data)
            
        except Exception as e:
            print(f"Error loading sample data: {str(e)}")
            # Return a small sample dataset as fallback
            return pd.DataFrame({
                'staff_id': ['1', '2', '3'],
                'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
                'latitude': [5.5826, 5.5926, 5.5726],
                'longitude': [-0.1435, -0.1335, -0.1535],
                'address': ['Accra', 'Accra', 'Accra']
            })

    def create_clusters(self, staff_data, grid_size=3, sigma=1.0, learning_rate=0.5):
        """Create clusters based on staff locations using SOM"""
        if staff_data is None or len(staff_data) == 0:
            return None
        
        try:
            # Calculate distances to office
            staff_data['distance_to_office'] = staff_data.apply(
                lambda row: geodesic(
                    (row['latitude'], row['longitude']),
                    (self.office_location['lat'], self.office_location['lon'])
                ).km,
                axis=1
            )
            
            # Prepare data for clustering
            locations = staff_data[['latitude', 'longitude', 'distance_to_office']].values
            normalized_data = self.scaler.fit_transform(locations)
            
            # Initialize and train SOM
            som = SOMCluster(
                input_len=3,
                grid_size=grid_size,
                sigma=sigma,
                learning_rate=learning_rate
            )
            
            som.train(normalized_data)
            
            # Assign clusters
            staff_data['cluster'] = [som.get_cluster(loc) for loc in normalized_data]
            
            # Handle small clusters
            self._handle_small_clusters(staff_data)
            
            return staff_data
            
        except Exception as e:
            print(f"Error in clustering: {str(e)}")
            return None

    def _handle_small_clusters(self, staff_data):
        """Handle clusters with fewer than minimum required passengers"""
        cluster_sizes = staff_data['cluster'].value_counts()
        small_clusters = cluster_sizes[cluster_sizes < self.MIN_PASSENGERS].index
        
        if len(small_clusters) > 0:
            for small_cluster in small_clusters:
                small_cluster_points = staff_data[staff_data['cluster'] == small_cluster]
                
                for idx, row in small_cluster_points.iterrows():
                    distances = []
                    for cluster_id in staff_data['cluster'].unique():
                        if cluster_id not in small_clusters:
                            cluster_points = staff_data[staff_data['cluster'] == cluster_id]
                            if not cluster_points.empty:
                                avg_dist = cluster_points.apply(
                                    lambda x: geodesic(
                                        (row['latitude'], row['longitude']),
                                        (x['latitude'], x['longitude'])
                                    ).km,
                                    axis=1
                                ).mean()
                                distances.append((cluster_id, avg_dist))
                    
                    if distances:
                        nearest_cluster = min(distances, key=lambda x: x[1])[0]
                        staff_data.at[idx, 'cluster'] = nearest_cluster

    def optimize_routes(self, staff_data):
        """Optimize routes within each cluster"""
        routes = defaultdict(list)
        route_counter = 0
        
        try:
            if 'distance_to_office' not in staff_data.columns:
                staff_data['distance_to_office'] = staff_data.apply(
                    lambda row: geodesic(
                        (row['latitude'], row['longitude']),
                        (self.office_location['lat'], self.office_location['lon'])
                    ).km,
                    axis=1
                )
            
            for cluster_id in staff_data['cluster'].unique():
                cluster_group = staff_data[staff_data['cluster'] == cluster_id].copy()
                
                while len(cluster_group) >= self.MIN_PASSENGERS:
                    current_route = []
                    remaining = cluster_group.copy()
                    
                    # Start with furthest person from office
                    start_person = remaining.nlargest(1, 'distance_to_office').iloc[0]
                    current_route.append(start_person.to_dict())
                    remaining = remaining.drop(start_person.name)
                    
                    while len(current_route) < self.MAX_PASSENGERS and not remaining.empty:
                        last_point = current_route[-1]
                        
                        remaining['temp_distance'] = remaining.apply(
                            lambda row: geodesic(
                                (last_point['latitude'], last_point['longitude']),
                                (row['latitude'], row['longitude'])
                            ).km,
                            axis=1
                        )
                        
                        next_person = remaining.nsmallest(1, 'temp_distance').iloc[0]
                        current_route.append(next_person.to_dict())
                        remaining = remaining.drop(next_person.name)
                        
                        if len(current_route) >= self.MIN_PASSENGERS:
                            break
                    
                    if len(current_route) >= self.MIN_PASSENGERS:
                        route_name = f'Route {route_counter + 1}'
                        routes[route_name] = current_route
                        route_counter += 1
                        assigned_ids = [p['staff_id'] for p in current_route]
                        cluster_group = cluster_group[~cluster_group['staff_id'].isin(assigned_ids)]
                    else:
                        break
                
                if len(cluster_group) > 0:
                    self._assign_remaining_staff(cluster_group, routes)
            
            return routes
            
        except Exception as e:
            print(f"Error in route optimization: {str(e)}")
            return {}

    def _assign_remaining_staff(self, remaining_staff, routes):
        """Assign remaining staff to existing routes"""
        if not isinstance(remaining_staff, pd.DataFrame):
            return
            
        for idx, row in remaining_staff.iterrows():
            best_route = None
            min_detour = float('inf')
            staff_dict = row.to_dict()
            
            for route_name, route_group in routes.items():
                if len(route_group) < self.MAX_PASSENGERS:
                    # Calculate current route distance
                    current_distance = self.calculate_route_metrics(route_group)[0]
                    
                    # Calculate new route distance with added staff member
                    test_route = route_group.copy()
                    test_route.append(staff_dict)
                    new_distance = self.calculate_route_metrics(test_route)[0]
                    
                    # Calculate detour distance
                    detour = new_distance - current_distance
                    
                    if detour < min_detour:
                        min_detour = detour
                        best_route = route_name
            
            if best_route:
                routes[best_route].append(staff_dict)

    def calculate_route_metrics(self, route):
        """Calculate total distance and cost for a route"""
        if not route:
            return 0, 0
        
        try:
            total_distance = 0
            
            # Convert route points to coordinate pairs
            points = [(p['latitude'], p['longitude']) for p in route]
            
            # Add office location as final destination
            points.append((self.office_location['lat'], self.office_location['lon']))
            
            # Calculate cumulative distance between all consecutive points
            for i in range(len(points) - 1):
                distance = geodesic(points[i], points[i + 1]).km
                total_distance += distance
            
            # Calculate total cost based on distance
            total_cost = total_distance * self.COST_PER_KM
            
            # Get route details from Google Maps for more accurate metrics
            try:
                waypoints = [{'lat': p[0], 'lng': p[1]} for p in points[1:-1]]
                route_data = self.get_route_directions(
                    f"{points[0][0]},{points[0][1]}",
                    f"{points[-1][0]},{points[-1][1]}",
                    waypoints=waypoints if waypoints else None
                )
                
                if route_data:
                    # Use Google Maps distance if available
                    total_distance = route_data['distance'] / 1000  # Convert meters to km
                    total_cost = total_distance * self.COST_PER_KM
            except Exception:
                # Fall back to geodesic distance if Google Maps fails
                pass
            
            return total_distance, total_cost
            
        except Exception as e:
            print(f"Error calculating route metrics: {str(e)}")
            return 0, 0

    def get_route_directions(self, origin, destination, waypoints=None):
        """Get route directions using Google Maps Directions API"""
        try:
            if waypoints:
                waypoints = [f"{point['lat']},{point['lng']}" for point in waypoints]
                directions = self.gmaps.directions(
                    origin,
                    destination,
                    waypoints=waypoints,
                    optimize_waypoints=True,
                    mode="driving",
                    departure_time=datetime.now()
                )
            else:
                directions = self.gmaps.directions(
                    origin,
                    destination,
                    mode="driving",
                    departure_time=datetime.now()
                )

            if directions:
                route = directions[0]
                route_polyline = route['overview_polyline']['points']
                duration = sum(leg['duration']['value'] for leg in route['legs'])
                distance = sum(leg['distance']['value'] for leg in route['legs'])
                
                return {
                    'polyline': route_polyline,
                    'duration': duration,
                    'distance': distance,
                    'directions': directions
                }
            return None
        except Exception as e:
            print(f"Error getting directions: {str(e)}")
            return None

    def calculate_total_metrics(self, routes):
        """Calculate total metrics for all routes"""
        try:
            total_metrics = {
                'total_distance': 0,
                'total_cost': 0,
                'total_passengers': 0,
                'number_of_routes': len(routes),
                'average_route_distance': 0,
                'average_route_cost': 0,
                'average_passengers_per_route': 0
            }
            
            for route_name, route in routes.items():
                distance, cost = self.calculate_route_metrics(route)
                total_metrics['total_distance'] += distance
                total_metrics['total_cost'] += cost
                total_metrics['total_passengers'] += len(route)
            
            if routes:
                total_metrics['average_route_distance'] = total_metrics['total_distance'] / len(routes)
                total_metrics['average_route_cost'] = total_metrics['total_cost'] / len(routes)
                total_metrics['average_passengers_per_route'] = total_metrics['total_passengers'] / len(routes)
            
            total_metrics['cost_per_passenger'] = (
                total_metrics['total_cost'] / total_metrics['total_passengers']
                if total_metrics['total_passengers'] > 0 else 0
            )
            
            return total_metrics
            
        except Exception as e:
            print(f"Error calculating total metrics: {str(e)}")
            return None

def main():
    # Example usage
    optimizer = StaffTransportOptimizer(google_maps_key=os.getenv("GOOGLE_MAPS_API_KEY"))
    
    # Load sample data
    staff_data = optimizer.load_sample_data()
    if staff_data is None:
        print(json.dumps({"error": "Failed to load sample data"}))
        return
    
    # Create clusters
    clustered_data = optimizer.create_clusters(
        staff_data,
        grid_size=3,
        sigma=1.0,
        learning_rate=0.5
    )
    
    if clustered_data is None:
        print(json.dumps({"error": "Clustering failed"}))
        return
    
    # Optimize routes
    routes = optimizer.optimize_routes(clustered_data)
    if not routes:
        print(json.dumps({"error": "Route optimization failed"}))
        return
    
    # Calculate metrics
    metrics = optimizer.calculate_total_metrics(routes)
    
    # Prepare route details
    route_details = {}
    for route_name, route in routes.items():
        distance, cost = optimizer.calculate_route_metrics(route)
        route_details[route_name] = {
            "distance": round(distance, 2),
            "cost": round(cost, 2),
            "passengers": [
                {
                    "name": passenger['name'],
                    "address": passenger['address'],
                    "staff_id": passenger['staff_id']
                }
                for passenger in route
            ]
        }
    
    # Prepare final output
    output = {
        "summary": {
            "total_routes": metrics['number_of_routes'],
            "total_passengers": metrics['total_passengers'],
            "total_distance": round(metrics['total_distance'], 2),
            "total_cost": round(metrics['total_cost'], 2),
            "average_route_distance": round(metrics['average_route_distance'], 2),
            "average_route_cost": round(metrics['average_route_cost'], 2),
            "average_passengers_per_route": round(metrics['average_passengers_per_route'], 1),
            "cost_per_passenger": round(metrics['cost_per_passenger'], 2)
        },
        "routes": route_details
    }
    
    # Print JSON output
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main() 