import misc.first as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
from geopy.distance import geodesic
from collections import defaultdict
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import googlemaps
import polyline
from folium.plugins import GroupedLayerControl
from dotenv import find_dotenv, load_dotenv
import os
import random
load_dotenv()

st.set_page_config(layout="wide")

class Particle:
    def __init__(self, dimensions, num_clusters, bounds=None):
        self.position = np.random.rand(dimensions, num_clusters)  # Centroids coordinates
        self.velocity = np.zeros((dimensions, num_clusters))
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.current_fitness = float('inf')
        self.bounds = bounds
        
    def update_velocity(self, global_best_position, w=0.5, c1=1.5, c2=1.5):
        """Update the velocity of the particle"""
        r1 = np.random.rand(self.position.shape[0], self.position.shape[1])
        r2 = np.random.rand(self.position.shape[0], self.position.shape[1])
        
        cognitive_velocity = c1 * r1 * (self.best_position - self.position)
        social_velocity = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity
    
    def update_position(self):
        """Update the position of the particle"""
        self.position = self.position + self.velocity
        
        # Apply bounds if provided
        if self.bounds is not None:
            for d in range(self.position.shape[0]):
                min_val, max_val = self.bounds[d]
                self.position[d] = np.clip(self.position[d], min_val, max_val)

    def calculate_fitness(self, data, metric='euclidean'):
        """Calculate the fitness of the particle (clustering quality)"""
        # Assign each data point to the closest centroid
        assignments = self.assign_clusters(data, metric)
        
        # Calculate the sum of squared distances within each cluster
        total_distance = 0
        for cluster_idx in range(self.position.shape[1]):
            cluster_points = data[assignments == cluster_idx]
            if len(cluster_points) > 0:
                centroid = self.position[:, cluster_idx]
                if metric == 'euclidean':
                    distances = np.sqrt(np.sum((cluster_points - centroid.reshape(1, -1))**2, axis=1))
                elif metric == 'manhattan':
                    distances = np.sum(np.abs(cluster_points - centroid.reshape(1, -1)), axis=1)
                total_distance += np.sum(distances)
        
        # Penalty for empty clusters
        unique_clusters = np.unique(assignments)
        empty_clusters_penalty = (self.position.shape[1] - len(unique_clusters)) * 1000
        
        # Penalty for clusters with too few points (less than MIN_PASSENGERS)
        small_clusters_penalty = 0
        for cluster_idx in range(self.position.shape[1]):
            cluster_size = np.sum(assignments == cluster_idx)
            if 0 < cluster_size < 3:  # MIN_PASSENGERS is assumed to be 3
                small_clusters_penalty += (3 - cluster_size) * 500
        
        return total_distance + empty_clusters_penalty + small_clusters_penalty
    
    def assign_clusters(self, data, metric='euclidean'):
        """Assign each data point to the closest centroid"""
        assignments = np.zeros(len(data), dtype=int)
        
        for i, point in enumerate(data):
            min_dist = float('inf')
            closest_cluster = 0
            
            for cluster_idx in range(self.position.shape[1]):
                centroid = self.position[:, cluster_idx]
                
                if metric == 'euclidean':
                    dist = np.sqrt(np.sum((point - centroid)**2))
                elif metric == 'manhattan':
                    dist = np.sum(np.abs(point - centroid))
                
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster = cluster_idx
            
            assignments[i] = closest_cluster
        
        return assignments

class PSOCluster:
    def __init__(self, num_clusters, dimensions, num_particles=30, max_iter=100, 
                 w=0.5, c1=1.5, c2=1.5, bounds=None):
        self.num_clusters = num_clusters
        self.dimensions = dimensions
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive weight
        self.c2 = c2  # Social weight
        self.bounds = bounds
        
        # Initialize particles
        self.particles = [Particle(dimensions, num_clusters, bounds) for _ in range(num_particles)]
        
        # Initialize global best
        self.global_best_position = np.random.rand(dimensions, num_clusters)
        self.global_best_fitness = float('inf')
    
    def optimize(self, data, metric='euclidean'):
        """Run the PSO algorithm to find optimal cluster centroids"""
        # Initial evaluation
        for particle in self.particles:
            particle.current_fitness = particle.calculate_fitness(data, metric)
            
            # Update particle's best
            if particle.current_fitness < particle.best_fitness:
                particle.best_fitness = particle.current_fitness
                particle.best_position = particle.position.copy()
            
            # Update global best
            if particle.best_fitness < self.global_best_fitness:
                self.global_best_fitness = particle.best_fitness
                self.global_best_position = particle.best_position.copy()
        
        # Optimization loop
        for _ in range(self.max_iter):
            for particle in self.particles:
                # Update velocity and position
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position()
                
                # Evaluate fitness
                particle.current_fitness = particle.calculate_fitness(data, metric)
                
                # Update particle's best
                if particle.current_fitness < particle.best_fitness:
                    particle.best_fitness = particle.current_fitness
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if particle.best_fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.best_fitness
                    self.global_best_position = particle.best_position.copy()
        
        return self.global_best_position
    
    def assign_clusters(self, data, metric='euclidean'):
        """Assign each data point to a cluster using the global best solution"""
        assignments = np.zeros(len(data), dtype=int)
        
        for i, point in enumerate(data):
            min_dist = float('inf')
            closest_cluster = 0
            
            for cluster_idx in range(self.global_best_position.shape[1]):
                centroid = self.global_best_position[:, cluster_idx]
                
                if metric == 'euclidean':
                    dist = np.sqrt(np.sum((point - centroid)**2))
                elif metric == 'manhattan':
                    dist = np.sum(np.abs(point - centroid))
                
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster = cluster_idx
            
            assignments[i] = closest_cluster
        
        return assignments

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
            invalid_coords = clean_df[clean_df['latitude'].isna() | clean_df['longitude'].isna()]
            if not invalid_coords.empty:
                st.warning(f"Removed {len(invalid_coords)} entries with invalid coordinates")
                clean_df = clean_df.dropna(subset=['latitude', 'longitude'])
            
            # Validate coordinate ranges for Ghana
            valid_lat_range = (4.5, 11.5)  # Ghana's latitude range
            valid_lon_range = (-3.5, 1.5)  # Ghana's longitude range
            
            # Create mask for valid coordinates
            coord_mask = (
                (clean_df['latitude'].between(*valid_lat_range)) &
                (clean_df['longitude'].between(*valid_lon_range))
            )
            
            invalid_range = clean_df[~coord_mask]
            if not invalid_range.empty:
                st.warning(f"Removed {len(invalid_range)} entries with coordinates outside Ghana")
                clean_df = clean_df[coord_mask]
            
            # Remove duplicates based on staff_id
            duplicates = clean_df[clean_df.duplicated(subset=['staff_id'], keep='first')]
            if not duplicates.empty:
                st.warning(f"Removed {len(duplicates)} duplicate staff entries")
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
            
            st.success(f"Successfully validated {len(clean_df)} staff records")
            return clean_df
            
        except Exception as e:
            st.error(f"Error validating staff data: {str(e)}")
            return None

    def load_sample_data(self):
        """Load sample staff location data for testing"""
        try:
            sample_data = pd.DataFrame({
                'staff_id': range(1, 21),
                'name': [f'Employee {i}' for i in range(1, 21)],
                'latitude': np.random.uniform(5.5526, 5.6126, 20),
                'longitude': np.random.uniform(-0.1735, -0.1135, 20),
                'address': [
                    'Adabraka', 'Osu', 'Cantonments', 'Airport Residential',
                    'East Legon', 'Spintex', 'Tema', 'Teshie', 'Labadi',
                    'Labone', 'Ridge', 'Roman Ridge', 'Dzorwulu', 'Abelemkpe',
                    'North Kaneshie', 'Dansoman', 'Mamprobi', 'Chorkor',
                    'Abeka', 'Achimota'
                ]
            })
            
            # Validate the sample data
            return self.validate_staff_data(sample_data)
            
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
            return None
            
    def create_clusters(self, staff_data, num_clusters=4, num_particles=30, max_iter=100):
        """Create clusters based on staff locations using PSO"""
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
            
            # Define bounds for centroids based on normalized data
            bounds = []
            for d in range(normalized_data.shape[1]):
                min_val = np.min(normalized_data[:, d])
                max_val = np.max(normalized_data[:, d])
                bounds.append((min_val, max_val))
            
            # Initialize and run PSO clustering
            pso = PSOCluster(
                num_clusters=num_clusters,
                dimensions=normalized_data.shape[1],
                num_particles=num_particles,
                max_iter=max_iter,
                bounds=bounds
            )
            
            pso.optimize(normalized_data)
            
            # Assign clusters
            staff_data['cluster'] = pso.assign_clusters(normalized_data)
            
            # Handle small clusters
            self._handle_small_clusters(staff_data)
            
            return staff_data
            
        except Exception as e:
            st.error(f"Error in clustering: {str(e)}")
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
        unassigned_staff = pd.DataFrame()
        
        try:
            if 'distance_to_office' not in staff_data.columns:
                staff_data['distance_to_office'] = staff_data.apply(
                    lambda row: geodesic(
                        (row['latitude'], row['longitude']),
                        (self.office_location['lat'], self.office_location['lon'])
                    ).km,
                    axis=1
                )
            
            # First pass: create optimal routes with MIN_PASSENGERS constraint
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
                
                # Try to assign remaining staff to existing routes
                if len(cluster_group) > 0:
                    before_assignment = len(cluster_group)
                    self._assign_remaining_staff(cluster_group, routes)
                    after_assignment = sum(1 for _, row in cluster_group.iterrows() 
                                         if row['staff_id'] not in [p['staff_id'] for route in routes.values() for p in route])
                    
                    # If any staff still unassigned, add them to our unassigned collection
                    if after_assignment > 0:
                        assigned_ids = [p['staff_id'] for route in routes.values() for p in route]
                        still_unassigned = cluster_group[~cluster_group['staff_id'].isin(assigned_ids)]
                        unassigned_staff = pd.concat([unassigned_staff, still_unassigned])
            
            # Second pass: create routes for any remaining unassigned staff
            if not unassigned_staff.empty:
                # Sort unassigned staff by distance to office
                unassigned_staff = unassigned_staff.sort_values('distance_to_office', ascending=False)
                
                current_route = []
                for idx, row in unassigned_staff.iterrows():
                    current_route.append(row.to_dict())
                    
                    # Create a new route when we hit MAX_PASSENGERS or at the end
                    if len(current_route) == self.MAX_PASSENGERS or idx == unassigned_staff.index[-1]:
                        route_name = f'Route {route_counter + 1}'
                        routes[route_name] = current_route
                        route_counter += 1
                        current_route = []
            
            return routes
            
        except Exception as e:
            st.error(f"Error in route optimization: {str(e)}")
            return {}

    def _assign_remaining_staff(self, remaining_staff, routes):
        """Assign remaining staff to existing routes"""
        if not isinstance(remaining_staff, pd.DataFrame):
            return
        
        # Keep track of which staff members have been assigned
        assigned_staff_ids = set()
            
        for idx, row in remaining_staff.iterrows():
            best_route = None
            min_detour = float('inf')
            staff_dict = row.to_dict()
            staff_id = staff_dict['staff_id']
            
            # Skip if this staff member has already been assigned
            if staff_id in assigned_staff_ids:
                continue
            
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
                assigned_staff_ids.add(staff_id)

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
            st.error(f"Error calculating route metrics: {str(e)}")
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
            st.error(f"Error getting directions: {str(e)}")
            return None

    def create_map(self, routes):
        """Create an interactive map with multiple layer controls and satellite imagery"""
        try:
            # Initialize base map
            m = folium.Map(
                location=[self.office_location['lat'], self.office_location['lon']],
                zoom_start=13,
                control_scale=True
            )
            
            # Add multiple tile layers
            folium.TileLayer(
                'cartodbpositron',
                name='Street Map',
                control=True
            ).add_to(m)
            
            folium.TileLayer(
                'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite View',
                control=True
            ).add_to(m)
            
            # Create office marker group
            office_group = folium.FeatureGroup(name='Office Location', show=True)
            folium.Marker(
                [self.office_location['lat'], self.office_location['lon']],
                popup=folium.Popup(
                    'Main Office',
                    max_width=200
                ),
                icon=folium.Icon(
                    color='red',
                    icon='building',
                    prefix='fa'
                ),
                tooltip="Office Location"
            ).add_to(office_group)
            office_group.add_to(m)
            
            # Define colors for routes
            colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
                     'violet', 'darkblue', 'darkgreen', 'cadetblue']
            
            # Create route groups
            for route_idx, (route_name, group) in enumerate(routes.items()):
                color = colors[route_idx % len(colors)]
                route_group = folium.FeatureGroup(
                    name=f"{route_name}",
                    show=True,
                    control=True
                )
                
                # Prepare waypoints for Google Maps
                waypoints = [
                    {'lat': staff['latitude'], 'lng': staff['longitude']}
                    for staff in group
                ]
                
                # Get route from Google Maps
                route_data = self.get_route_directions(
                    f"{waypoints[0]['lat']},{waypoints[0]['lng']}",
                    f"{self.office_location['lat']},{self.office_location['lon']}",
                    waypoints[1:] if len(waypoints) > 1 else None
                )
                
                if route_data:
                    # Add route polyline
                    route_coords = polyline.decode(route_data['polyline'])
                    
                    # Create route path with popup
                    route_line = folium.PolyLine(
                        route_coords,
                        weight=4,
                        color=color,
                        opacity=0.8,
                        popup=folium.Popup(
                            f"""
                            <div style='font-family: Arial; font-size: 12px;'>
                                <b>{route_name}</b><br>
                                Distance: {route_data['distance']/1000:.2f} km<br>
                                Duration: {route_data['duration']/60:.0f} min<br>
                                Passengers: {len(group)}
                            </div>
                            """,
                            max_width=200
                        )
                    )
                    route_line.add_to(route_group)
                    
                    # Add staff markers
                    for idx, staff in enumerate(group, 1):
                        # Create detailed popup content
                        popup_content = f"""
                        <div style='font-family: Arial; font-size: 12px;'>
                            <b>{staff['name']}</b><br>
                            Address: {staff['address']}<br>
                            Stop #{idx}<br>
                            Distance to office: {staff['distance_to_office']:.2f} km<br>
                            Pick-up order: {idx} of {len(group)}
                        </div>
                        """
                        
                        # Add staff marker
                        folium.CircleMarker(
                            location=[staff['latitude'], staff['longitude']],
                            radius=8,
                            popup=folium.Popup(
                                popup_content,
                                max_width=200
                            ),
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.7,
                            weight=2,
                            tooltip=f"Stop #{idx}: {staff['name']}"
                        ).add_to(route_group)
                
                route_group.add_to(m)
            
            # Add layer controls
            folium.LayerControl(
                position='topright',
                collapsed=False,
                autoZIndex=True
            ).add_to(m)
            
            # Add fullscreen control
            folium.plugins.Fullscreen(
                position='topleft',
                title='Fullscreen',
                title_cancel='Exit fullscreen',
                force_separate_button=True
            ).add_to(m)
            
            # Add search control
            folium.plugins.Search(
                layer=office_group,
                search_label='name',
                position='topleft'
            ).add_to(m)
            
            return m
            
        except Exception as e:
            st.error(f"Error creating map: {str(e)}")
            return None

    def get_route_summary(self, route):
        """
        Get a detailed summary of route metrics
        
        Args:
            route (list): List of dictionaries containing staff information
            
        Returns:
            dict: Dictionary containing route metrics
        """
        try:
            distance, cost = self.calculate_route_metrics(route)
            
            return {
                'total_distance': distance,
                'total_cost': cost,
                'passenger_count': len(route),
                'cost_per_passenger': cost / len(route) if route else 0,
                'distance_per_passenger': distance / len(route) if route else 0,
                'start_point': route[0]['address'] if route else None,
                'end_point': 'Office',
                'stops': len(route)
            }
        except Exception as e:
            st.error(f"Error generating route summary: {str(e)}")
            return None

    def calculate_total_metrics(self, routes):
        """
        Calculate total metrics for all routes
        
        Args:
            routes (dict): Dictionary of routes
            
        Returns:
            dict: Dictionary containing total metrics
        """
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
            st.error(f"Error calculating total metrics: {str(e)}")
            return None

    def format_metrics(self, metrics):
        """
        Format metrics for display
        
        Args:
            metrics (dict): Dictionary containing metrics
            
        Returns:
            dict: Dictionary containing formatted metrics
        """
        try:
            formatted = {}
            for key, value in metrics.items():
                if 'distance' in key.lower():
                    formatted[key] = f"{value:.2f} km"
                elif 'cost' in key.lower():
                    formatted[key] = f"GHC{value:.2f}"
                elif 'average' in key.lower():
                    if 'distance' in key.lower():
                        formatted[key] = f"{value:.2f} km"
                    elif 'cost' in key.lower():
                        formatted[key] = f"GHC{value:.2f}"
                    else:
                        formatted[key] = f"{value:.2f}"
                else:
                    formatted[key] = value
            return formatted
            
        except Exception as e:
            st.error(f"Error formatting metrics: {str(e)}")
            return None

    def create_metrics_summary(self, routes):
        """
        Create a complete metrics summary for display
        
        Args:
            routes (dict): Dictionary of routes
            
        Returns:
            dict: Dictionary containing formatted metrics summary
        """
        try:
            metrics = self.calculate_total_metrics(routes)
            if metrics:
                formatted_metrics = self.format_metrics(metrics)
                
                summary = {
                    'Overview': {
                        'Total Routes': metrics['number_of_routes'],
                        'Total Passengers': metrics['total_passengers'],
                        'Total Distance': formatted_metrics['total_distance'],
                        'Total Cost': formatted_metrics['total_cost']
                    },
                    'Averages': {
                        'Average Route Distance': formatted_metrics['average_route_distance'],
                        'Average Route Cost': formatted_metrics['average_route_cost'],
                        'Average Passengers per Route': f"{metrics['average_passengers_per_route']:.1f}",
                        'Cost per Passenger': formatted_metrics['cost_per_passenger']
                    },
                    'Routes': {}
                }
                
                for route_name, route in routes.items():
                    route_summary = self.get_route_summary(route)
                    if route_summary:
                        formatted_route_summary = self.format_metrics(route_summary)
                        summary['Routes'][route_name] = formatted_route_summary
                
                return summary
            return None
            
        except Exception as e:
            st.error(f"Error creating metrics summary: {str(e)}")
            return None

# Part 3: UI Helper Functions
def load_css():
    st.markdown(
        """
        <meta charset="UTF-8">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.19.1/css/mdb.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
        <link rel="icon" href="https://www.4th-ir.com/favicon.ico">
        
        <title>4thir-POC-repo</title>
        <meta name="title" content="4thir-POC-repo" />
        <meta name="description" content="view our proof of concepts" />

        <!-- Open Graph / Facebook -->
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://4thir-poc-repositoty.streamlit.app/" />
        <meta property="og:title" content="4thir-POC-repo" />
        <meta property="og:description" content="view our proof of concepts" />
        <meta property="og:image" content="https://www.4th-ir.com/favicon.ico" />

        <!-- Twitter -->
        <meta property="twitter:card" content="summary_large_image" />
        <meta property="twitter:url" content="https://4thir-poc-repositoty.streamlit.app/" />
        <meta property="twitter:title" content="4thir-POC-repo" />
        <meta property="twitter:description" content="view our proof of concepts" />
        <meta property="twitter:image" content="https://www.4th-ir.com/favicon.ico" />

        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        """,
        unsafe_allow_html=True,
    )

    
    st.markdown(
        """
        <style>
            header {visibility: hidden;}
            .main {
                margin-top: -20px;
                padding-top: 10px;
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .navbar {
                padding: 1rem;
                margin-bottom: 2rem;
                background-color: #4267B2;
                color: white;
            }
            .card {
                padding: 1rem;
                margin-bottom: 1rem;
                transition: transform 0.2s;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .card:hover {
                transform: scale(1.02);
            }
            .metric-card {
               
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem;
                
            }
            .search-box {
                margin-bottom: 1rem;
                padding: 0.5rem;
                border-radius: 4px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def create_navbar():
    st.markdown(
        """
       <nav class="navbar fixed-top navbar-expand-lg navbar-dark bg-white text-bold shadow-sm">
            <a class="navbar-brand text-primary" href="#" target="_blank">
                <img src="https://cdn.bio.link/uploads/profile_pictures/2022-01-27/Q0mOblteBj6VooKH4zNCa9zKD5JHkVnM.png" alt="4th-ir logo" style='width:50px'>
               Ride Router - PSO Edition
            </a>
        </nav>
        """,
        unsafe_allow_html=True
    )

def show_metrics_dashboard(metrics):
    st.markdown("""
        <div class="card" style="padding: 1rem;  border-radius: 8px; margin-bottom: 1rem;">
            <h3 style="margin-bottom: 1rem;">Route Metrics Dashboard</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
    """, unsafe_allow_html=True)
    
    for key, value in metrics.items():
        st.markdown(f"""
            <div class="metric-card card">
                <h4>{key.replace('_', ' ').title()}</h4>
                <p style="font-size: 1.5rem; font-weight: bold;">{value}</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

# Part 4: Main Application
def main():
    load_css()
    create_navbar()
    
    # Initialize session state
    if 'optimization_done' not in st.session_state:
        st.session_state.optimization_done = False
    if 'staff_data' not in st.session_state:
        st.session_state.staff_data = None
    if 'routes' not in st.session_state:
        st.session_state.routes = None

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("1. Data Input")
        data_option = st.radio(
            "Choose data input method",
            ["Upload CSV", "Use Sample Data"]
        )
        
        optimizer = StaffTransportOptimizer(google_maps_key=os.getenv("GOOGLE_MAPS_API_KEY"))
        
        if data_option == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload staff locations CSV",
                type=["csv"],
                help="""
                Required columns:
                - staff_id (unique identifier)
                - name (staff name)
                - latitude (valid coordinate)
                - longitude (valid coordinate)
                - address (location description)
                """
            )
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.staff_data = optimizer.validate_staff_data(df)
                    st.success("Data validated successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.staff_data = None
        else:
            if st.button("Load Sample Data"):
                st.session_state.staff_data = optimizer.load_sample_data()
                st.success("Sample data loaded successfully!")
        
        st.subheader("2. PSO Parameters")
        num_clusters = st.slider(
            "Number of Clusters",
            min_value=2,
            max_value=8,
            value=4,
            help="Controls the number of potential clusters"
        )
        
        num_particles = st.slider(
            "Number of Particles",
            min_value=10,
            max_value=50,
            value=30,
            help="Number of particles in the swarm (more particles = more exploration)"
        )
        
        max_iter = st.slider(
            "Maximum Iterations",
            min_value=50,
            max_value=200,
            value=100,
            step=10,
            help="Maximum number of iterations for PSO algorithm"
        )
        
        if st.button("Optimize Routes", type="primary"):
            if st.session_state.staff_data is not None:
                with st.spinner("Optimizing routes with PSO algorithm..."):
                    try:
                        clustered_data = optimizer.create_clusters(
                            st.session_state.staff_data,
                            num_clusters=num_clusters,
                            num_particles=num_particles,
                            max_iter=max_iter
                        )
                        
                        if clustered_data is not None:
                            st.session_state.routes = optimizer.optimize_routes(clustered_data)
                            if st.session_state.routes:
                                st.session_state.optimization_done = True
                                st.success("Routes optimized successfully with PSO!")
                            else:
                                st.error("Route optimization failed. Try different parameters.")
                        else:
                            st.error("Clustering failed. Try different parameters.")
                    except Exception as e:
                        st.error(f"Optimization error: {str(e)}")
            else:
                st.warning("Please upload valid staff data first.")

    # Main content area
    if st.session_state.staff_data is not None:
        col1, col2, = st.columns([2, 1])
        
        with col1:
            if st.session_state.optimization_done:
                st.subheader("Route Map")
                m = optimizer.create_map(st.session_state.routes)
                if m is not None:
                    st_folium(m, width=800, height=600)
                    
                    st.info("""
                    **Map Controls:**
                    - Use the layer control (top right) to toggle routes and map type
                    - Click on markers for detailed information
                    - Zoom in/out using the mouse wheel or +/- buttons
                    """)
        
        with col2:
            st.subheader("Staff Directory")
            search_term = st.text_input("Search staff by name or address")
            
            display_df = st.session_state.staff_data[['name', 'address','latitude','longitude']].copy()
            if search_term:
                mask = (
                    display_df['name'].str.contains(search_term, case=False) |
                    display_df['address'].str.contains(search_term, case=False)
                    
                )
                display_df = display_df[mask]
            
            st.dataframe(display_df, height=300)
            
            if st.session_state.optimization_done:
                st.subheader("Route Details")
                
                metrics = {
                    'total_distance': 0,
                    'total_cost': 0,
                    'total_duration': 0
                }
                
                for route_name, route in st.session_state.routes.items():
                    with st.expander(f"{route_name}"):
                        distance, cost = optimizer.calculate_route_metrics(route)
                        metrics['total_distance'] += distance
                        metrics['total_cost'] += cost
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Distance", f"{distance:.2f} km")
                        with col2:
                            st.metric("Cost", f"GHC{cost:.2f}")
                        
                        st.dataframe(
                            pd.DataFrame(route)[['name', 'address', 'distance_to_office']],
                            height=200
                        )
                
                show_metrics_dashboard({
                    'Total Distance': f"{metrics['total_distance']:.2f} km",
                    'Total Cost': f"GHC{metrics['total_cost']:.2f}",
                    'Number of Routes': len(st.session_state.routes),
                    'Average Cost/Route': f"GHC{metrics['total_cost']/len(st.session_state.routes):.2f}"
                })

if __name__ == "__main__":
    main()