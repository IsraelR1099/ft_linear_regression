import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse


class LinearRegression:
    def __init__(self, x, y):
        self.data = x
        self.label = y
        self.theta0 = 0  # Intercept
        self.theta1 = 0  # Slope
        self.n = len(x)

    def fit(self, epochs, lr):
        # Gradient Descent
        for i in range(epochs):
            y_pred = self.theta1 * self.data + self.theta0
            error = y_pred - self.label
            d_theta0 = (1 / self.n) * np.sum(error)
            d_theta1 = (1 / self.n) * np.sum(error * self.data)
            self.theta0 -= lr * d_theta0
            self.theta1 -= lr * d_theta1

    def predict(self, input):
        y_pred = self.theta1 * input + self.theta0
        return y_pred

    def save_model(self, filename="model.csv"):
        with open(filename, "w") as f:
            f.write(f"{self.theta0},{self.theta1}")


def normalize(value):
    """
    Returns normalized value along with mean and standard deviation
    """
    value_mean = np.mean(value)
    value_std = np.std(value)
    return (value - value_mean) / value_std, value_mean, value_std


def denormalize_theta(theta0, theta1, x_mean, x_std, y_mean, y_std):
    """
    Denormalizes theta0 and theta1
    """
    theta1_denorm = theta1 * y_std / x_std
    theta0_denorm = y_mean + y_std * (theta0 - theta1 * x_mean / x_std)
    return theta0_denorm, theta1_denorm


def load_data():
    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        exit(1)
    x = np.array(df.iloc[:, 0])
    y = np.array(df.iloc[:, 1])
    return x, y


def train_model(x_norm, y_norm, epochs=1000, lr=0.01):
    model = LinearRegression(x_norm, y_norm)
    model.fit(epochs=epochs, lr=lr)
    return model


def plot_data(x, y):
    plt.scatter(x, y, color="blue", label="Data points")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (€)")
    plt.title("Car Prive vs Mileage")
    plt.legend()
    plt.show()


def plot_regression_line(x, y, theta0, theta1):
    y_pred = theta0 + theta1 * x
    plt.scatter(x, y, color="blue", label="Data")
    plt.plot(x, y_pred, color="red", label="Fitted line")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (€)")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.show()


def plot_residuals(x, y_true, y_pred):
    residuals = y_true - y_pred
    plt.scatter(
        x, residuals, color="purple", alpha=0.7
    )
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residual Analysis")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Residuals (Error €)")
    plt.show()


def calculate_r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a linear regression model on car price vs mileage data."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of epochs for training (default: 1000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate."
    )
    parser.add_argument(
        "--metric",
        choices=["r2", "mse", "both"],
        default="both",
        help="Evaluation metric to display (default: both)"
    )
    parser.add_argument(
        "--compare-lr",
        action="store_true",
        help="Compare multiple learning rates visually."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    x, y = load_data()
    plot_data(x, y)
    x_norm, x_mean, x_std = normalize(x)
    y_norm, y_mean, y_std = normalize(y)
    if args.compare_lr:
        plt.scatter(
            x, y, color="gray", alpha=0.5, label="Data points"
        )
        for lr in [0.001, 0.01, 0.05, 0.1]:
            model = LinearRegression(x_norm, y_norm)
            model.fit(epochs=args.epochs, lr=lr)
            theta0, theta1 = denormalize_theta(
                model.theta0, model.theta1, x_mean, x_std, y_mean, y_std
            )
            y_pred = theta0 + theta1 * x
            plt.plot(x, y_pred, label=f"lr={lr}")
        plt.xlabel("Mileage (km)")
        plt.ylabel("Price (€)")
        plt.title("Learning Rate Comparison")
        plt.legend()
        plt.show()
        exit(0)
    model = LinearRegression(x_norm, y_norm)
    model.fit(epochs=args.epochs, lr=args.lr)
    theta0, theta1 = denormalize_theta(
        model.theta0, model.theta1, x_mean, x_std, y_mean, y_std
    )
    model.theta0, model.theta1 = theta0, theta1
    model.save_model()
    print("✅ Model saved to model.csv")
    print(f"   theta0 = {theta0:.0f}, theta1 = {theta1:.0f}")

    y_pred = theta0 + theta1 * x
    if args.metric in ("r2", "both"):
        r2 = calculate_r2_score(y, y_pred)
        print(f"📊 R² Score: {r2:.4f}")
    if args.metric in ("mse", "both"):
        mse = calculate_mse(y, y_pred)
        print(f"📉 Mean Squared Error: {mse:.2f}")

    # Visualizations
    plot_regression_line(x, y, theta0, theta1)
    plot_residuals(x, y, y_pred)
