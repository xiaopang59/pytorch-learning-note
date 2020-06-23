import numpy as np

# y = wx + b
def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, w_current, point, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(point))
    for i in range(0, len(point)):
        x = point[i, 0]
        y = point[i, 1]
        b_gradient += -(2/N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return new_b, new_w

def gradient_descent_runner(points, starting_b, starting_w,
                            learning_rate, num_iterator):
    b = starting_b
    w = starting_w
    for i in range(num_iterator):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return b, w

def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0   # initial y-intercept guess
    initial_w = 0   # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_error_for_line_given_points(initial_b, initial_w, points)))
    print("Running....")
    b, w = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterators b = {1}, w = {2}, error = {3}"
          .format(num_iterations, b, w,
                  compute_error_for_line_given_points(b, w, points))
          )

if __name__ == "__main__":
    run()