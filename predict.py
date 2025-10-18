def load_model(filename="model.csv"):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            theta0, theta1 = map(float, file.readline().strip().split(","))
        return theta0, theta1
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        exit(1)


def predict_price(mileage, theta0, theta1):
    if mileage < 0:
        raise ValueError("Mileage cannot be negative.")
    if mileage > 1e6:
        print("⚠️ Warning: The entered mileage is unrealistic for a car."
                "The prediction may not be accurate.")
    estimated_price = theta0 + theta1 * mileage
    if estimated_price < 0:
        print("⚠️ Warning: The predicted price is negative."
                "The prediction may not be accurate.")
        estimated_price = 0.0
    return estimated_price


if __name__ == '__main__':
    theta0, theta1 = load_model()
    mileage = input("Please enter the mileage of the car: ")
    try:
        mileage = float(mileage)
    except ValueError:
        print("Invalid input. Please enter a number.")
        exit(1)
    predicted_price = predict_price(mileage, theta0, theta1)
    print(
        f"The estimated price of a car with {mileage} km is {predicted_price:.2f} €.")
