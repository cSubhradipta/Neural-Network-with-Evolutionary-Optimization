import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("sample_data/weather_classification_data.csv")

print("Initial Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

df.fillna(df.mode().iloc[0], inplace=True)

target_column = "Weather Type"

y_raw = df[target_column]
X_raw = df.drop(columns=[target_column])

categorical_cols = X_raw.select_dtypes(include=['object']).columns

X_processed = pd.get_dummies(X_raw, columns=categorical_cols)

y_labels = pd.factorize(y_raw)[0]

num_classes = len(np.unique(y_labels))

def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

y_processed = one_hot(y_labels, num_classes)

print("Number of classes:", num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed.values,
    y_processed,
    test_size=0.2,
    random_state=42,
    stratify=y_labels
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_main, X_val, y_train_main, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=42,
    stratify=np.argmax(y_train, axis=1)
)

print("Train main:", X_train_main.shape)
print("Validation:", X_val.shape)

print("Final Train Shape:", X_train.shape)
print("Final Test Shape:", X_test.shape)
print("Target shape:", y_train.shape)

class MLP_GA:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.h1 = 128
        self.h2 = 64
        self.output_size = output_size

        self.num_weights = (
            input_size * self.h1 +
            self.h1 +
            self.h1 * self.h2 +
            self.h2 +
            self.h2 * output_size +
            output_size
        )

    def decode(self, chromosome):
        idx = 0

        W1 = chromosome[idx:idx+self.input_size*self.h1].reshape(self.input_size, self.h1)
        idx += self.input_size*self.h1
        b1 = chromosome[idx:idx+self.h1]
        idx += self.h1

        W2 = chromosome[idx:idx+self.h1*self.h2].reshape(self.h1, self.h2)
        idx += self.h1*self.h2
        b2 = chromosome[idx:idx+self.h2]
        idx += self.h2

        W3 = chromosome[idx:idx+self.h2*self.output_size].reshape(self.h2, self.output_size)
        idx += self.h2*self.output_size
        b3 = chromosome[idx:idx+self.output_size]

        return W1, b1, W2, b2, W3, b3

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X, chromosome):
        W1, b1, W2, b2, W3, b3 = self.decode(chromosome)

        a1 = self.relu(X @ W1 + b1)
        a2 = self.relu(a1 @ W2 + b2)
        out = self.softmax(a2 @ W3 + b3)

        return out

class GeneticAlgo:
    def __init__(self, model, pop_size=120, generations=300):
        self.model = model
        self.pop_size = pop_size
        self.generations = generations
        self.elite = int(0.05 * pop_size)

        self.population = np.random.randn(pop_size, model.num_weights) * 0.5

    def fitness(self, chromosome, X_val, y_val, lambda_reg=0.0005):

        preds = self.model.forward(X_val, chromosome)

        loss = -np.mean(np.sum(y_val * np.log(preds + 1e-8), axis=1))
        loss += lambda_reg * np.sum(chromosome**2)

        fitness = 1 / (1 + loss)

        return fitness


    def tournament(self, fitness_scores, k=5):
        idx = np.random.choice(self.pop_size, k)
        return self.population[idx[np.argmax(fitness_scores[idx])]]

    def evolve(self, X_train, y_train, X_val, y_val):

        for gen in range(self.generations):

            fitness_scores = np.array([
                self.fitness(ind, X_val, y_val)
                for ind in self.population
            ])

            sorted_idx = np.argsort(fitness_scores)[::-1]
            self.population = self.population[sorted_idx]
            fitness_scores = fitness_scores[sorted_idx]

            new_population = self.population[:self.elite].copy()

            mutation_rate = 0.25 * (1 - gen / self.generations)
            sigma = 0.3 * (1 - gen / self.generations)

            while len(new_population) < self.pop_size:

                parent1 = self.tournament(fitness_scores)
                parent2 = self.tournament(fitness_scores)

                mask = np.random.rand(len(parent1)) > 0.5
                child = np.where(mask, parent1, parent2)

                mutation_mask = np.random.rand(len(child)) < mutation_rate
                child[mutation_mask] += np.random.normal(
                    0, sigma, np.sum(mutation_mask)
                )

                child = np.clip(child, -5, 5)

                new_population = np.vstack((new_population, child))

            self.population = new_population

            if gen % 20 == 0:
                print(f"Generation {gen}, \tFitness value: {fitness_scores[0]:.6f}")

        return self.population[0]

model = MLP_GA(
    input_size=X_train.shape[1],
    output_size=y_train.shape[1]
)

ga = GeneticAlgo(
    model,
    pop_size=120,
    generations=300
)

best_weights = ga.evolve(X_train_main, y_train_main, X_val, y_val)

test_preds = model.forward(X_test, best_weights)

pred_classes = np.argmax(test_preds, axis=1)
true_classes = np.argmax(y_test, axis=1)

test_accuracy = np.mean(pred_classes == true_classes)

print("Final Test Accuracy:", test_accuracy)
