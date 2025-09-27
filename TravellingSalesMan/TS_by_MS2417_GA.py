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

# Class to store and calculate fitness of a route
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = self.calculate_total_distance()
        self.fitness = 1 / self.distance if self.distance > 0 else float('inf')  # avoid div by zero

    def calculate_total_distance(self):
        distance = 0
        for i in range(len(self.route)):
            start_city = self.route[i]
            next_city = self.route[(i + 1) % len(self.route)]
            distance += start_city.distance_to(next_city)
        return distance

# Main class implementing Genetic Algorithm for TSP with elitism
class GeneticAlgorithmTSP:
    def __init__(self, cities, population_size=100, generations=500, mutation_rate=0.01, elite_size=5):
        self.cities = cities
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.elite_size = elite_size  # How many best routes to keep unchanged per generation
        self.population = []
        self.fitness_results = []
        self.fitness_scores = []
        self.avg_fitness_progress = []

    def create_route(self):
        return random.sample(self.cities, len(self.cities))

    def initial_population(self):
        self.population = [self.create_route() for _ in range(self.population_size)]

    def rank_population(self):
        self.fitness_results = [(i, Fitness(route)) for i, route in enumerate(self.population)]
        # Sort routes by fitness descending
        self.fitness_results.sort(key=lambda x: x[1].fitness, reverse=True)
        self.fitness_scores = [(i, fitness_obj.fitness) for i, fitness_obj in self.fitness_results]

    def selection(self):
        df = sum(fitness for i, fitness in self.fitness_scores)
        prob = [fitness / df for i, fitness in self.fitness_scores]
        selected_indices = np.random.choice([i for i, _ in self.fitness_scores],
                                            size=self.population_size - self.elite_size,
                                            p=prob)
        return [self.population[index] for index in selected_indices]

    def crossover(self, parent1, parent2):
        start = random.randint(0, len(parent1) - 1)
        end = random.randint(start, len(parent1) - 1)
        child_p1 = parent1[start:end]
        child_p2 = [city for city in parent2 if city not in child_p1]
        return child_p1 + child_p2

    def mutate(self, route):
        for swapped in range(len(route)):
            if random.random() < self.mutation_rate:
                swap_with = int(random.random() * len(route))
                route[swapped], route[swap_with] = route[swap_with], route[swapped]
        return route

    def next_generation(self):
        self.rank_population()
        # Elitism: keep best routes unchanged
        elites = [self.population[i] for i, _ in self.fitness_results[:self.elite_size]]
        selected = self.selection()
        pool = random.sample(selected, len(selected))
        children = [self.crossover(pool[i], pool[(i + 1) % len(pool)]) for i in range(len(pool))]
        mutated_children = [self.mutate(child) for child in children]
        self.population = elites + mutated_children  # Elites + new population

    def evolve(self):
        self.initial_population()

        for generation in range(self.generations):
            self.next_generation()
            total_fitness = sum(fitness_obj.fitness for index, fitness_obj in self.fitness_results)
            avg_fitness = total_fitness / self.population_size
            self.avg_fitness_progress.append(avg_fitness)

            if generation % 100 == 0 or generation == self.generations - 1:
                best_route_idx, best_fitness_obj = self.fitness_results[0]
                best_route = self.population[best_route_idx]
                print(f"Generation {generation}: Best Distance = {best_fitness_obj.distance:.2f}")
                self.plot_route(best_route, generation)


    def plot_route(self, route, generation):
        plt.figure(figsize=(10, 6))
        plt.title(f'TSP Route at Generation {generation}')
        x = [city.x for city in route] + [route[0].x]
        y = [city.y for city in route] + [route[0].y]
        plt.plot(x, y, 'b-', marker='o')
        for city in route:
            plt.text(city.x, city.y, city.name, fontsize=9, ha='right', va='bottom')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()


def readcities():
    cities = []
    geolocator = Nominatim(user_agent="fss_app")
    with open("India_cities.txt") as file:
        for line in file:
            city = line.strip()
            if not city:
                continue
            location_query = city + ", India"
            try:
                pt = geolocator.geocode(location_query, timeout=100)
                if pt is None:
                    print(f"Warning: Could not find location for {city}")
                    continue
                x = round(pt.longitude, 2)
                y = round(pt.latitude, 2)
                print(f"City = {city}, Latitude = {pt.latitude}, Longitude = {pt.longitude}")
                cities.append(City(city, x, y))
            except Exception as e:
                print(f"Error locating city {city}: {e}")
                continue
    return cities

# Main execution
cities = readcities()
ga_tsp = GeneticAlgorithmTSP(cities, population_size=200, generations=500, mutation_rate=0.02, elite_size=5)
ga_tsp.evolve()
