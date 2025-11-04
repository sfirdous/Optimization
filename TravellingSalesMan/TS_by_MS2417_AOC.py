import numpy as np
import matplotlib.pyplot as plt
import random
from geopy.geocoders import Nominatim

# Class to represent city name and coordinates
class City:
    def __init__(self, name, x, y):
        self.x = x
        self.y = y
        self.name = name

    def distance_to(self, city):
        return np.sqrt((self.x - city.x)**2 + (self.y - city.y)**2)

# Class implementing Ant Colony Optimization for TSP
class AntColonyTSP:
    def __init__(self, cities, n_ants=30, n_iterations=100, alpha=1.0, beta=2.5,evaporation_rate=0.5, Q=100):
        
        self.cities = cities
        self.n_cities = len(cities)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        
        # Build distance matrix
        self.distances = self.build_distance_matrix()
        
        # Initialize pheromone matrix
        self.pheromones = np.ones((self.n_cities, self.n_cities))
        
        # Best solution tracking
        self.best_route = None
        self.best_distance = float('inf')
        self.distance_progress = []
        
    def build_distance_matrix(self):
        distance_matrix = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j:
                    distance_matrix[i][j] = self.cities[i].distance_to(self.cities[j])
        return distance_matrix
    
    def calculate_probabilities(self, current_city_idx, unvisited_indices):
        pheromone = np.array([self.pheromones[current_city_idx][idx] for idx in unvisited_indices])
        distance = np.array([self.distances[current_city_idx][idx] for idx in unvisited_indices])
        
        # Avoid division by zero
        distance = np.where(distance == 0, 1e-10, distance)
        
        # Attractiveness = pheromone^alpha * (1/distance)^beta
        attractiveness = (pheromone ** self.alpha) * ((1.0 / distance) ** self.beta)
        probabilities = attractiveness / attractiveness.sum()
        
        return probabilities
    
    def construct_solution(self):
        start_idx = random.randint(0, self.n_cities - 1)
        route_indices = [start_idx]
        unvisited = list(range(self.n_cities))
        unvisited.remove(start_idx)
        
        current_idx = start_idx
        
        while unvisited:
            probabilities = self.calculate_probabilities(current_idx, unvisited)
            next_idx = np.random.choice(unvisited, p=probabilities)
            route_indices.append(next_idx)
            unvisited.remove(next_idx)
            current_idx = next_idx
        
        # Convert indices to City objects
        route = [self.cities[idx] for idx in route_indices]
        return route
    
    def calculate_route_distance(self, route):
        distance = 0
        for i in range(len(route)):
            start_city = route[i]
            next_city = route[(i + 1) % len(route)]
            distance += start_city.distance_to(next_city)
        return distance
    
    def update_pheromones(self, all_routes, all_distances):
        # Evaporation
        self.pheromones *= (1 - self.evaporation_rate)
        
        # Add new pheromones
        for route, distance in zip(all_routes, all_distances):
            pheromone_deposit = self.Q / distance
            for i in range(len(route)):
                city1_idx = self.cities.index(route[i])
                city2_idx = self.cities.index(route[(i + 1) % len(route)])
                self.pheromones[city1_idx][city2_idx] += pheromone_deposit
                self.pheromones[city2_idx][city1_idx] += pheromone_deposit
    
    def iterate(self):
        print(f"\nStarting Ant Colony Optimization")
        print(f"Cities: {self.n_cities}, Ants: {self.n_ants}, Iterations: {self.n_iterations}")
        print("=" * 70)
        
        for iteration in range(self.n_iterations):
            all_routes = []
            all_distances = []
            
            # Construct solutions for all ants
            for ant in range(self.n_ants):
                route = self.construct_solution()
                distance = self.calculate_route_distance(route)
                all_routes.append(route)
                all_distances.append(distance)
                
                # Update best solution
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_route = route
            
            # Update pheromones
            self.update_pheromones(all_routes, all_distances)
            
            # Track progress
            self.distance_progress.append(self.best_distance)
            
            # Display progress and plot at intervals
            if iteration % 20 == 0 or iteration == self.n_iterations - 1:
                print(f"Iteration {iteration}: Best Distance = {self.best_distance:.2f}")
                self.plot_route(self.best_route, iteration)
        
        print("=" * 70)
        print(f"Optimization Complete! Final Best Distance: {self.best_distance:.2f}\n")
        self.print_route(self.best_route)
    
    def print_route(self, route):
        print("\nBest Route Found:")
        for i, city in enumerate(route):
            print(f"  {i + 1}. {city.name}")
        print(f"  {len(route) + 1}. {route[0].name} (return to start)")
        print(f"\nTotal Distance: {self.best_distance:.2f}")
    
    def plot_route(self, route, iteration):
        """Plot the route on a graph"""
        plt.figure(figsize=(12, 8))
        plt.title(f'TSP Route at Iteration {iteration} (Distance: {self.calculate_route_distance(route):.2f})', 
                  fontsize=14, fontweight='bold')
        
        x = [city.x for city in route] + [route[0].x]
        y = [city.y for city in route] + [route[0].y]
        
        # Plot the route
        plt.plot(x, y, 'b-', linewidth=2, alpha=0.7)
        plt.plot(x, y, 'ro', markersize=10, markeredgecolor='darkred', markeredgewidth=2)
        
        # Add city names
        for city in route:
            plt.text(city.x, city.y, f'  {city.name}', fontsize=9, ha='left', va='bottom', 
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def readcities(filename):
    """Read cities from file and get their coordinates"""
    cities = []
    geolocator = Nominatim(user_agent="aco_tsp_app")
    
    try:
        with open(filename) as file:
            for line in file:
                city_name = line.strip()
                if not city_name:
                    continue
                location_query = city_name + ", India"
                try:
                    pt = geolocator.geocode(location_query, timeout=100)
                    if pt is None:
                        print(f"Warning: Could not find location for {city_name}")
                        continue
                    x = round(pt.longitude, 2)
                    y = round(pt.latitude, 2)
                    print(f"City = {city_name}, Latitude = {pt.latitude}, Longitude = {pt.longitude}")
                    cities.append(City(city_name, x, y))
                except Exception as e:
                    print(f"Error locating city {city_name}: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: 'India_cities.txt' file not found!")
        print("Please create a file named 'India_cities.txt' or try with complete path of the file.")
        return []
    
    return cities


# Main execution
if __name__ == "__main__":
    print("Reading cities from India_cities.txt...")
    cities = readcities("D:\Optimization\TravellingSalesMan\India_cities.txt")
    
    if len(cities) < 2:
        print("Error: Need at least 2 cities to solve TSP!")
    else:
        print(f"\nSuccessfully loaded {len(cities)} cities.")
        
        # Initialize and run ACO
        aco_tsp = AntColonyTSP(
            cities=cities, 
            n_ants=30, 
            n_iterations=100, 
            alpha=1.0, 
            beta=2.5, 
            evaporation_rate=0.5, 
            Q=100
        )
        
        # Solve the TSP
        aco_tsp.iterate()