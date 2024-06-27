import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return 0.5 / (1 + np.exp(-20 * (0.5 - x))) + 0.5

# Generate x values
x = np.linspace(0.0, 1.0, 400)

# Generate y values using the function
y = f(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='f(x)')

# Add labels and title
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Graph of the function f(x)')
plt.legend()

# Show the plot
plt.grid(True)
plt.savefig("temp.pdf")