# Polynomial-Guessing

## About

Given an input data file which consists of the a set of points (x,y) such that `y = f(x) + ε` where ε is the error, we try to estimate f(x) assuming that f(x) is a polynomial in x of unknown degree in the range [1,10]  
The resulting polynomial is displayed in the file `Output.txt`

## Algorithmic Specifications

The following algorithm is a simplistic linear regression model which constructs input data using powers of the data point x.      
Full batch size gradient descent has been used to optimize the weights.    
To add diversity to the model, 5 different error functions are used to verify the final estimate.  
The user has also been given an option to plot the data set and the the predicted polynomial on a same graph to compare the result with the data.  

## Tools Used

* Python
* Numpy
* Matplotlib

## Usage

The data file needs to be store in the `Data` folder. A specific format needs to be followed.  
The data file should have 2 columns, first column containing x co-ordinates and second column containing y co-ordinates.   
Moreover, the data file should be headed by the row `x y`  
For reference, a sample data file has been stored in the `Data` folder.  

To execute the code, use `python Poly.py -df input_file.txt`  
If you wish to plot the data as well add the command `--plot`  
If you wish to use a specific error function you can add the command `-lf num` where num is an integer in the range [1,5]

You can use `python Poly.py -h` in case you need any help. This would display the following:  

```
-h, --help            Display this help message and exit
-lf, --loss_function  Enter the loss function for gradient 1: Mean Square, 2: Mean Absolute, 3: Mean Root, 4: Cross Entropy, 5: Log-Cosh
-df, --data_file      Enter the address of the data file to be used
-pl, --plot           If you want to plot the polynomial against points
```

