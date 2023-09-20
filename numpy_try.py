# numPy Arrays
import numpy as np
my_list = [1,2,3] # list 
np.array(my_list)
my_list == [1,2,3]; np.array(my_list) == [1,2,3] # numpy array allows entry-wise operation

my_matrix = [[1,2,3],[4,5,6],[7,8,9]] # list of lists = matrix
np.array(my_matrix)
np.eye(4) # diag(4) identity matrix

np.arange(0,10) # 0:9
np.arange(0,11,2) # seq(0,11,by=2)
np.linspace(0,10,3) # seq(0, 9, length.out=3)

np.zeros(3) # zeros
np.ones(3) # ones

## random
np.random.rand(2, 5) # uniformly distributed matrix
np.random.randn(2, 5) # normally distributed matrix
np.random.randint(1,10, 10) # sample w replacement
np.random.choice(my_list, 3, replace=False) # sample from a given list w/ replacement

## reshape
ranarr = np.random.randint(0,50,10)
ranarr = ranarr.reshape(2, 5); ranarr; ranarr.shape
ranarr.max(); ranarr.argmax(); ranarr.min(); ranarr.argmin();

## indexing and selection
arr = np.arange(0,11)
arr[1:5] #Get values in a range
arr[0:5]=100; arr # broadcasting (entry-wise operation)
arr1=arr[0:2]
arr1[:]=5; arr1; arr # arr1 points arr[0:2]. Data are not copied, it uses pointer to avoid memory and data structure problems
arr1=arr1.copy() # allows to copy itselfs
arr1[:]=6; arr1; arr # arr1 points arr[0:2]. Data are not copied, it uses pointer to avoid memory and data structure problems

## indexing a 2D array
arr_2d = np.array(([5,10,15],[20,25,30],[35,40,45])); arr_2d
arr_2d[1] # indexing row
arr_2d[2][2]; arr_2d[2,2] # same result of indexing individual value
arr_2d[:2,1:] # topright corner

## use T/F
arr_2d[:,np.arange(3)%2==0] # even columns

arr+arr; arr-arr; arr*arr; arr/arr ## Arithmetic
np.sqrt(arr); np.exp(arr); np.sin(arr); np.log(arr) ## universal array functions
np.max(arr); arr.max() ## universal array function vs attribute function
