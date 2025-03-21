import numpy as np

def compute_discriminant(alphas, y, x_values):
    def kernel(x_i, x):
        return (x_i * x + 1) ** 2
    
    coeff_x2 = 0
    coeff_x = 0
    coeff_const = 0
    
    for i in range(len(alphas)):
        if alphas[i] != 0:
            contrib = alphas[i] * y[i]
            
            # Expand the squared term (x_i * x + 1)^2 = x_i^2 * x^2 + 2 * x_i * x + 1
            x_i = x_values[i]
            coeff_x2 += contrib * (x_i ** 2)
            coeff_x += contrib * (2 * x_i)
            coeff_const += contrib * 1
    
    return round(coeff_x2, 2), round(coeff_x, 2), round(coeff_const, 2)

# Given values
alphas = [0, 2.5, 0, 7.333, 4.833]
y = [1, 1, -1, -1, 1]
x_values = [1, 2, 4, 5, 6]

# Compute the coefficients
a, b, c = compute_discriminant(alphas, y, x_values)

# Print the final equation
print(f"g(x) = {a:+} x^2 {b:+} x {c:+}")
