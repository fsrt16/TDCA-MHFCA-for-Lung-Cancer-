import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from zoofs.baseoptimizationalgorithm import BaseOptimizationAlgorithm

# -------------------------------
# Define the Cheetah Hunting Algorithm (CHA)
# -------------------------------

class CheetahHuntingAlgorithm(BaseOptimizationAlgorithm):
    def __init__(self, population_size=20, n_iterations=30, minimize=True):
        super().__init__(population_size, n_iterations, minimize)
        self.loss_history = []  # Store loss values for convergence plotting

    def optimize(self, fitness_function, model, X_train, y_train, X_valid, y_valid, bounds):
        """Run CHA optimization."""
        num_params = len(bounds)
        population = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (self.population_size, num_params))
        
        for _ in range(self.n_iterations):
            fitness_values = self.evaluate_fitness(fitness_function, model, X_train, y_train, X_valid, y_valid, population, bounds)
            self.loss_history.append(np.min(fitness_values))  # Store best loss
            
            # Apply Cheetah-inspired updates (mutation, recombination)
            new_population = self.cheetah_update(population, fitness_values, bounds)
            population = new_population

        best_idx = np.argmin(fitness_values)
        return population[best_idx], fitness_values[best_idx]

    def evaluate_fitness(self, fitness_function, model, X_train, y_train, X_valid, y_valid, population, bounds):
        """Evaluate loss for each individual."""
        fitness_values = []
        num_features = X_train.shape[1]
        
        for individual in population:
            params = {f'param_{i}': individual[i] for i in range(len(bounds) - num_features)}
            feature_mask = np.array(individual[len(bounds) - num_features:], dtype=bool)
            X_train_selected = X_train[:, feature_mask]
            X_valid_selected = X_valid[:, feature_mask]
            
            loss = fitness_function(model, X_train_selected, y_train, X_valid_selected, y_valid, params)
            fitness_values.append(loss)
        
        return np.array(fitness_values)

    def cheetah_update(self, population, fitness_values, bounds):
        """Apply CHA-based updates (mutation & exploration)."""
        new_population = np.copy(population)
        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx]

        for i in range(self.population_size):
            perturbation = np.random.uniform(-0.1, 0.1, size=len(bounds))
            new_population[i] = best_solution + perturbation  # Move toward best
            new_population[i] = np.clip(new_population[i], [b[0] for b in bounds], [b[1] for b in bounds])

        return new_population

# -------------------------------
# Define Objective Function
# -------------------------------
def objective_function(model, X_train, y_train, X_valid, y_valid, params):
    model.set_params(**params)
    model.fit(X_train, y_train)
    return log_loss(y_valid, model.predict_proba(X_valid))

# -------------------------------
# Load Dataset & Preprocess
# -------------------------------
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Define Search Space (Hyperparameters + Feature Selection)
# -------------------------------
num_features = X.shape[1]
bounds = [(0.01, 0.3), (10, 100)] + [(0, 1)] * num_features  # Hyperparameters + Feature Mask

# -------------------------------
# Initialize CHA Optimizer
# -------------------------------
cha_optimizer = CheetahHuntingAlgorithm(population_size=20, n_iterations=30)

# -------------------------------
# Initialize LightGBM Model
# -------------------------------
lgb_model = lgb.LGBMClassifier()

# -------------------------------
# Run CHA Optimization
# -------------------------------
best_params, best_loss = cha_optimizer.optimize(
    fitness_function=objective_function,
    model=lgb_model,
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid,
    y_valid=y_valid,
    bounds=bounds
)

# -------------------------------
# Extract Selected Features
# -------------------------------
best_mask = best_params[len(bounds) - num_features:]
selected_features = np.where(np.array(best_mask) == 1)[0]

# -------------------------------
# Print Results
# -------------------------------
print("Best Hyperparameters:", best_params[:len(bounds) - num_features])
print("Best Log Loss:", best_loss)
print("Selected Features:", selected_features)

# -------------------------------
# Plot Convergence
# -------------------------------
def plot_convergence(loss_values):
    """Plot convergence of CHA optimization."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(loss_values)), loss_values, marker='o', linestyle='-', color='b', label="Log Loss")
    plt.title("CHA Optimization Convergence", fontsize=14, fontweight='bold')
    plt.xlabel("Iteration", fontsize=12, fontweight='bold')
    plt.ylabel("Log Loss", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()

if hasattr(cha_optimizer, "loss_history"):
    plot_convergence(cha_optimizer.loss_history)
