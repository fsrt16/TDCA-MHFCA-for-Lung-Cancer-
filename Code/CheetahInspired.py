import numpy as np
from sklearn.metrics import log_loss

class CheetahHuntingAlgorithm:
    """Cheetah Hunting Algorithm (CHA) for hyperparameter optimization and feature selection."""
    
    def __init__(self, fitness_function, max_iter=20, population_size=20, minimize=True):
        self.fitness_function = fitness_function
        self.max_iter = max_iter
        self.population_size = population_size
        self.minimize = minimize
        self.best_params = None
        self.best_fitness = float('inf') if minimize else float('-inf')
        self.best_features = None
    
    def initialize_population(self, num_params):
        """Initialize population randomly."""
        return np.random.rand(self.population_size, num_params)

    def evaluate_fitness(self, model, X_train, y_train, X_valid, y_valid, population):
        """Evaluate fitness for each individual in the population."""
        return np.array([
            self.fitness_function(model, X_train, y_train, X_valid, y_valid, 
                                  {f'param_{i}': params[i] for i in range(len(params))})
            for params in population
        ])
    
    def update_position(self, position, velocity):
        """Update position."""
        return position + velocity
    
    def select_features(self, params):
        """Select feature indices based on optimized parameters."""
        selected_features = [i for i, val in enumerate(params) if val > 0.5]
        return selected_features
    
    def fit(self, model, X_train, y_train, X_valid, y_valid, feature_names):
        """Optimize hyperparameters using the Cheetah Hunting Algorithm and select features."""
        num_params = len(feature_names)
        
        # Step 1: Initialize population and velocity
        population = self.initialize_population(num_params)
        velocity = np.zeros_like(population)
        
        # Step 2: Evaluate initial fitness
        fitness_values = self.evaluate_fitness(model, X_train, y_train, X_valid, y_valid, population)
        
        # Step 3: Find the best solution
        best_index = np.argmin(fitness_values) if self.minimize else np.argmax(fitness_values)
        self.best_params = population[best_index].copy()
        self.best_fitness = fitness_values[best_index]
        self.best_features = self.select_features(self.best_params)
        
        # Optimization loop
        for _ in range(self.max_iter):
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)  # Generate random weights
                velocity[i] = (0.5 * velocity[i]) + (1.5 * r1 * (self.best_params - population[i])) + (1.5 * r2 * (self.best_params - population[i]))
                population[i] = self.update_position(population[i], velocity[i])
                
                # Evaluate new fitness
                new_fitness = self.fitness_function(model, X_train, y_train, X_valid, y_valid, 
                                                    {f'param_{j}': population[i, j] for j in range(num_params)})
                print('MINIMUM IN ITERATION')
                print(self.minimize, new_fitness)
                # Update global best
                if (self.minimize and new_fitness < self.best_fitness) or (not self.minimize and new_fitness > self.best_fitness):
                    self.best_params = population[i].copy()
                    self.best_fitness = new_fitness
                    self.best_features = self.select_features(self.best_params)
                print('MINIMUM IN BEST')
                print(self.best_features, self.best_fitness)
        
        return self.best_params, self.best_fitness, self.best_features
