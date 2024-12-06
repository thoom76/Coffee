import numpy as np
import matplotlib.pyplot as plt

optimal_bitterness = {
    "median": 5.6,
    "n_range": 0.6
} 
optimal_sourness = {
    "median": 5,
    "n_range": 0.5
}
get_bitterness_score = lambda n: 6 + (1 - (abs(optimal_bitterness["median"] - abs(n)) / optimal_bitterness["n_range"])) * 4 if abs(optimal_bitterness["median"] - abs(n)) < optimal_bitterness["n_range"] else 5 - np.clip((1 + abs(optimal_bitterness["median"] - abs(n)) ** 1.7), 0, 5)
get_sourness_score = lambda n: 6 + (1 - (abs(optimal_sourness["median"] - abs(n)) / optimal_sourness["n_range"])) * 4 if abs(optimal_sourness["median"] - abs(n)) < optimal_sourness["n_range"] else 5 - np.clip((1 + abs(optimal_sourness["median"] - abs(n)) ** 1.7), 0, 5)
get_end_score = lambda n: (get_sourness_score(n) + get_bitterness_score(n)) / 2

if __name__ == "__main__":
    # Step 1: Define the range for x (e.g., 100 evenly spaced points between 0 and 10)
    x = np.linspace(0, 10, 1000)

    # Compute y values
    y1 = [get_bitterness_score(n) for n in x]
    y2 = [get_sourness_score(n) for n in x]
    y3 = [get_end_score(n) for n in x]


    # Step 3: Plot the values
    plt.plot(x, y1, label="bitterness", color="blue")
    plt.plot(x, y2, label="sourness", color="purple")
    plt.plot(x, y3, label="endscore", color="green")

    # Add labels, legend, and grid
    plt.title("Function Visualization")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()

    # Display the plot
    plt.show()
