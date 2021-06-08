import random
import numpy
import matplotlib.pyplot as plt
import math
import argparse


parser = argparse.ArgumentParser(description='Guess the polynomial from the data containing points of the for (x,y)')


parser.add_argument('-lf','--loss_function', type=int, metavar='',help="Enter the loss function for gradient 1: Mean Square, 2: Mean Absolute, 3: Mean Root, 4: Cross Entropy, 5: Log-Cosh")
parser.add_argument('-df','--data_file', metavar='',help="Enter the address of the data file to be used", required=True)
parser.add_argument('-pl','--plot',help="If you want to plot the polynomial against points", action="store_true")
args = parser.parse_args()

# Constants
#Types of learning model
LEARNING_RATE_LIST = [[10**(-2*i) for i in range(1,11)],[10**(-1*i) for i in range(1,11)],[10,1,1,10**(-1),10**(-1),10**(-2),10**(-2),10**(-3),10**(-4),10**(-5),10**(-5)],[10**(-2),10**(-3),10**(-4),10**(-6),10**(-7),10**(-9),10**(-10),10**(-12),10**(-13),10**(-14)],[10**(-1*i) for i in range(1,11)]]

SQUARED_TYPE_VALUE = 1
ABSOLUTE_TYPE_VALUE = 2
ROOT_TYPE_VALUE = 3
CROSS_TYPE_VALUE = 4
HYPERBOLIC_TYPE_VALUE = 5

INITIALIZATION_VALUE = [0,0]
LN_2 = 0.693
INPUT_FILE = "Data\\" + args.data_file
OUTPUT_FILE = "Output.txt"
PLOT = args.plot # Do you wanna plot the resultant polynomial with data points
CREATE_FILE = True # Create a file containg data using the polynomial.

# Parameters 
ITERATIONS = 50000
DEGREE_LIMIT = 11
REGULARISATION = True
SHUFFLE = True
REGULARISATION_PARAMETER = 0
TYPE = args.loss_function if(args.loss_function) else 5

file_to_read = open(INPUT_FILE,'r')
t = 0
for line in file_to_read:
    if (len(line.split())):
        t = t + 1

DATASET_SIZE = t - 1
TRAINING_SET_SIZE = 9*(DATASET_SIZE//10)
TESTING_SET_SIZE = DATASET_SIZE - TRAINING_SET_SIZE

def regularize_gradient(gradient, W, regularisation_parameter):
    if REGULARISATION:
        gradient = gradient + regularisation_parameter*W
        gradient[0] = gradient[0] - regularisation_parameter*W[0]
    return gradient

def regularize_error(error,W,regularisation_parameter):
    if REGULARISATION:
       reg_term = (regularisation_parameter/2)*(numpy.sum(numpy.multiply(W,W)) - (W[0]*W[0]))
       error = error + reg_term
    return error

#Difference error
def difference_error_function(X,W,y):
    N = len(y)
    XW = numpy.dot(X,W)
    XW_y = numpy.subtract(XW,y)
    error = numpy.sum(XW_y)
    return error/N


# log(cosh()) error
def hyperbolicLog_error_function(X,W,y,regularisation_parameter):
    N = len(y)
    XW = numpy.dot(X,W)
    XW_y = numpy.subtract(XW,y)
    error_list = numpy.zeros(len(y))
    for idx in range(len(y)):
        if abs(XW_y[idx]) > 100:
            error_list[idx] = abs(XW_y[idx]) - LN_2
        else:
            error_list[idx] = math.log(math.cosh(XW_y[idx]))
    error = numpy.sum(error_list)
    error = regularize_error(error,W,regularisation_parameter)
    return error/N


# root error function (summation (|h(xi) - yi|)^(1/2))/N
def root_error_function(X,W,y,regularisation_parameter):
    N = len(y)
    XW = numpy.dot(X,W)
    XW_y = numpy.subtract(XW,y)
    XW_y = numpy.abs(XW_y)
    rXW_y = numpy.sqrt(XW_y)
    error = numpy.sum(rXW_y)
    error = regularize_error(error,W,regularisation_parameter)
    return error/N

#absolute error function (Summation(|h(xi) - yi|))/N
def absolute_error_function(X, W, y, regularisation_parameter):
    N = len(y)
    XW = numpy.dot(X,W)
    XW_y = numpy.subtract(XW,y)
    XW_y = numpy.abs(XW_y)
    error = numpy.sum(XW_y)
    error = regularize_error(error,W,regularisation_parameter)
    return error/N

# Squared error function (Summation(h(xi) - yi)^2)/N
def squared_error_function(X, W, y, regularisation_parameter):
    N = len(y)
    XW = numpy.dot(X,W)
    XW_y = numpy.subtract(XW,y)
    XW_yT = numpy.transpose(XW_y)
    error = numpy.dot(XW_yT,XW_y)
    error = regularize_error(error,W,regularisation_parameter)
    return error/N

def cross_entropy_error(X, W, y, regularisation_parameter):
    N = len(y)
    XW = numpy.dot(X,W)
    XW_y = numpy.subtract(XW,y)
    temp3 = numpy.multiply(XW_y,XW_y)
    for i in range(0,N):
        if(temp3[i] <= 100):
            temp3[i] = math.log(1 + math.exp(temp3[i]))   
        error = numpy.sum(temp3)
    error = regularize_error(error,W,regularisation_parameter)
    return error/N


# error function with type of error function parameter
def error_function(target_output_list, X_matrix, weights, type_value, regularisation_parameter):
    if type_value == SQUARED_TYPE_VALUE:
        return squared_error_function(X_matrix, weights, target_output_list, regularisation_parameter)
    elif type_value == ABSOLUTE_TYPE_VALUE:
        return absolute_error_function(X_matrix, weights, target_output_list, regularisation_parameter)
    elif type_value == ROOT_TYPE_VALUE:
        return root_error_function(X_matrix, weights, target_output_list, regularisation_parameter)
    elif type_value == CROSS_TYPE_VALUE:
        return cross_entropy_error(X_matrix, weights, target_output_list, regularisation_parameter)
    elif type_value == HYPERBOLIC_TYPE_VALUE:
        return hyperbolicLog_error_function(X_matrix, weights, target_output_list, regularisation_parameter)
    else:
        raise ValueError("This is not a valid input for type of error function")

# Gradient descent for squared error function
def update_weights(X, W, y, learning_rate,type_value, regularisation_parameter):
    if type_value == SQUARED_TYPE_VALUE:
        return update_square(X,W,y,learning_rate,regularisation_parameter)
    elif type_value == ABSOLUTE_TYPE_VALUE:
        return update_absolute(X,W,y,learning_rate,regularisation_parameter)
    elif type_value == ROOT_TYPE_VALUE:
        return update_root(X,W,y,learning_rate,regularisation_parameter)
    elif type_value == CROSS_TYPE_VALUE:
        return update_cross(X,W,y,learning_rate,regularisation_parameter)
    elif type_value == HYPERBOLIC_TYPE_VALUE:
        return update_hyperbolic(X,W,y,learning_rate,regularisation_parameter)
    else:
        raise ValueError("This is not a valid input for type of error function")


def update_square(X, W, y, learning_rate,regularisation_parameter):
    N = len(X)
    X_T = numpy.transpose(X)
    XTX = numpy.dot(X_T,X)
    XTXW = numpy.dot(XTX,W)
    XTy = numpy.dot(X_T,y)
    gradient = 2*numpy.subtract(XTXW,XTy)
    gradient = regularize_gradient(gradient,W,regularisation_parameter)
    gradient = learning_rate*gradient/N
    W1 = numpy.subtract(W,gradient)
    return W1

def update_absolute(X, W, y, learning_rate,regularisation_parameter):
    N = len(y)
    X_T = numpy.transpose(X)
    XW = numpy.dot(X,W)
    XW_y = numpy.subtract(XW,y)
    SI = numpy.sign(XW_y)
    gradient = numpy.dot(X_T,SI)
    gradient = regularize_gradient(gradient,W,regularisation_parameter)
    gradient = learning_rate*gradient/N
    W1 = numpy.subtract(W,gradient)
    return W1

def update_root(X,W,y, learning_rate,regularisation_parameter):
    N = len(y)
    X_T = numpy.transpose(X)
    XW = numpy.dot(X,W)
    XW_y = numpy.subtract(XW,y)
    XW_y = numpy.abs(XW_y)
    rXW_y = numpy.sqrt(XW_y)
    irXW_y = numpy.reciprocal(rXW_y)/2
    SI = numpy.sign(XW_y)
    irXW_y = numpy.multiply(irXW_y,SI)
    gradient = numpy.dot(X_T,irXW_y)
    gradient = regularize_gradient(gradient,W,regularisation_parameter)
    gradient = learning_rate*gradient/N
    W1 = numpy.subtract(W,gradient)
    return W1

def update_cross(X,W,y, learning_rate,regularisation_parameter):
    N = len(y)
    X_T = numpy.transpose(X)
    XW = numpy.dot(X,W)
    temp = numpy.subtract(XW,y)
    for i in range(0,N):
        if(temp[i]>10):
            temp[i] = 2*temp[i]
        else:
            z = temp[i]
            z = z/(1+math.exp(-1*z*z))
            temp[i] = 2*z 
    gradient = numpy.dot(X_T,temp)
    gradient = regularize_gradient(gradient,W,regularisation_parameter)
    gradient = learning_rate*gradient/N
    W1 = numpy.subtract(W,gradient)
    return W1

def update_hyperbolic(X,W,y, learning_rate,regularisation_parameter):
    N = len(y)
    X_T = numpy.transpose(X)
    XW = numpy.dot(X,W)
    XW_y = numpy.subtract(XW,y)
    temp = numpy.tanh(XW_y)
    gradient = numpy.dot(X_T,temp)
    gradient = regularize_gradient(gradient,W,regularisation_parameter)
    gradient = learning_rate*gradient/N
    W1 = numpy.subtract(W,gradient)
    return W1

#Get output from weights and input values
def create_output_list(weights, X_matrix):
    return numpy.dot(X_matrix,weights)
    
# Extract target function data:
def initialize_target(data):
    target_data = [x[1] for x in data]
    target_data = numpy.array(target_data)
    return target_data

#Extract X matrix
def initialize_X(data, degree):
    X_matrix = [[x[0]**i for i in range(degree+1)] for x in data]
    X_matrix = numpy.array(X_matrix)
    return X_matrix

# The main linear regression algorithm
def main_algo(training_data, testing_data, iterations, learning_rate, degree, type_value, regularisation_parameter):
    weights = [random.randint(0,5) for i in range(degree+1)]
    weights = numpy.array(weights)
    X_matrix_train = initialize_X(training_data,degree)
    X_matrix_test = initialize_X(testing_data, degree)
    target_data_train = initialize_target(training_data)
    target_data_test = initialize_target(testing_data)
    for i in range(iterations):
        weights = update_weights(X_matrix_train,weights,target_data_train,learning_rate,type_value, regularisation_parameter)
    error_train = error_function(target_data_train, X_matrix_train, weights, type_value, regularisation_parameter)
    if(TRAINING_SET_SIZE != DATASET_SIZE):
        error_test = error_function(target_data_test, X_matrix_test, weights, type_value, regularisation_parameter)
    else: 
        error_test = error_train
    return([error_test,error_train,weights])

# getting polynomial function to plot in graph
def PolyCoefficients(x, coeffs):
    o = len(coeffs)
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y

# Function to plot graph with scatter plot and predicted polynomial
def plot_result(coefficients, x, y):
    plt.scatter(x,y)
    starting_point = int(min(x)) -1
    ending_point = int(max(x)) + 1
    x1 = numpy.linspace(starting_point,ending_point,100)
    plt.plot(x1,PolyCoefficients(x1,coefficients))
    plt.show()
    plt.close()

# Creates an output file and prints test error and training error along with iterations and degree
def create_outputfile(paras):
    outputFile = open(OUTPUT_FILE, 'a')
    outputFile.write( "Degree : " + str(paras[0]) + " Testing Error : " + str(paras[1]) + " Training Error : " + str(paras[2]) + "\n")
    polynomial = ""
    for i in range(len(paras[3])):
        polynomial = str(paras[3][i]) + "x^" + str(i) + " + " + polynomial
    polynomial = polynomial[:-2]
    outputFile.write(" The Estimated polynomial : " + polynomial)
    outputFile.write("\n")
    outputFile.close()



# Reading Data from the file
fileData = open(INPUT_FILE , "r")
dataSet = [INITIALIZATION_VALUE for i in range(100)]
i = 0
for line in fileData:
    if(i != 0):
        dataSet[i-1] = line.split()
        dataSet[i-1][0] = float(dataSet[i-1][0])
        dataSet[i-1][1] = float(dataSet[i-1][1])
    i = i + 1
fileData.close()

if SHUFFLE:
    numpy.random.shuffle(dataSet)

#Extracting Training set from the main data set
TrainingSet = [INITIALIZATION_VALUE for i in range(TRAINING_SET_SIZE)] 
idx = 0
while idx < TRAINING_SET_SIZE:
    TrainingSet[idx] = dataSet[idx]
    idx = idx + 1

#Extracting Testing set from the main data set
TestingSet = [INITIALIZATION_VALUE for i in range(TESTING_SET_SIZE)]  
idx = TRAINING_SET_SIZE
while idx < DATASET_SIZE:
    TestingSet[idx-TRAINING_SET_SIZE] = dataSet[idx]
    idx = idx + 1

# For scatter plot on graph of the given data
outputList = [x[1] for x in dataSet]
inputList = [x[0] for x in dataSet]

# Driver code:
if CREATE_FILE:
    f = open(OUTPUT_FILE, 'w')
    f.close()
idx = 0


if __name__ == '__main__':
    best = -1
    paras = []
    for DEGREE in range(1,DEGREE_LIMIT):
        LEARNING_RATE = LEARNING_RATE_LIST[TYPE-1][DEGREE-1]
        result = main_algo(TrainingSet, TestingSet, ITERATIONS, LEARNING_RATE, DEGREE, TYPE, REGULARISATION_PARAMETER)
        if(best == -1 or best > result[0]):
            best = result[0]
            paras = [DEGREE,result[0],result[1],result[2]]
    if CREATE_FILE:
        create_outputfile(paras)
    if PLOT:
        plot_result(paras[3], inputList, outputList)