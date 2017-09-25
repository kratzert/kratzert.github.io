---
layout: post
title: 'Numba series part 2: Custom data types and parallelization'
date: '2017-09-25 15:04'
excerpt: >-
  Here we will focus on how we can use custom data types inside of Numba
  optimized functions as well as parallelization.
comments: true
---

This is the second part of my little series about the Numba library. This time we will take a look on how we can use custom data types inside of functions we like to get optimized by Numba. Out-of-the-box Numba can handle scalars and n-dimensional Numpy arrays as input. Tuples and lists are also supported but lists for example have to be strictly homogeneous (even integers and floats in one list are not supported). As we have seen in the [last part](https://kratzert.github.io/2017/09/21/numba-series-part-1-the-jit-decorator-and-some-more-numba-basics.html) Pythons dictionaries are not supported. I personally like dictonaries and their key-indexing and find them really useful in many occasions. Luckily there is a way, how we can use something similar inside a Numba optimized function: `numpy.dtype()`
To be honest, I never used this feature of the amazing Numpy library before, but here it comes quite handy. If you, as me before, don't know it already: You can use `numpy.dtype()` to specify custom types with key/value pairs, quite similar to dictionaries and these can be used inside Numba optimized functions.

In the same occasion we will take a first look on parallelization.

Let's have a look how we can do this.


```python
import timeit
import numpy as np

from numba import jit, prange
```

Let's use the hydrological ABC-Model again (see the [introductory article](https://kratzert.github.io/2017/09/12/introduction-to-the-numba-library.html) for a more detailed explanation), we tried to use with Python dictionaries without success in last part. Remember that this model has three different model parameters (a, b, c) we need to pass for a simulation.

First we have to create a custom type with Numpy ([here](https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.dtypes.html) is the official documentation for this feature). For this we can pass a list of tuples to the `np.dtype()` function, where each tuple specifies the name of a field and the data type of the elements in this field. Here we will create three fields, one for each parameter.


```python
abc_dtype = np.dtype([('a', np.float64),
                      ('b', np.float64),
                      ('c', np.float64)])

abc_dtype
```

    dtype([('a', '<f8'), ('b', '<f8'), ('c', '<f8')])


This can now be used like any of the build-in dtypes for any of the Numpy functions. Like creating an empty array filled with zeros.


```python
arr = np.zeros(3, dtype=abc_dtype)
arr
```

    array([( 0.,  0.,  0.), ( 0.,  0.,  0.), ( 0.,  0.,  0.)],
          dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])


We now have created an array of length three filled with zeros, where each entry consists of three values. As for dictionaries we can access any of the fields with their name and specify the index as for any Numpy array. E.g. the first value of the parameter `a` can be accessed by:


```python
arr['a'][0] = 1
arr
```

    array([( 1.,  0.,  0.), ( 0.,  0.,  0.), ( 0.,  0.,  0.)],
          dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])


Okay, now let's adapt the ABC-Model a little bit, so that we can pass multiple sets of model parameters as input and receive one time series of simulated stream flow for each parameter set. This also sounds like we could parallelize this job, since each simulation can be made independent of the others. The documentation says [here](http://numba.pydata.org/numba-doc/0.35.0/user/parallel.html) that the parallelization feature is still experimental, so this might change in future but for version 0.35 we have two different methods how we can tell Numba to parallelize parts of our code - implicit and explicit.

- `implicit` means, that we just pass another flag to the `@jit` decorator, namely `parallel=True`.
- for-loops can be marked `explicitly` to be parallelized by using another function of the Numba library - the `prange` function. This can be used like Pythons `range` but tells Numba that this loop can be parallelized. If you want to use `prange` you have to make sure, that there are no cross iteration dependencies.

Okay now lets implement four different versions of the ABC-Model, which all take multiple sets of parameters as inputs using our custom dtype. As reference we implement one version, that won't be optimized by Numba, to compare the speed to a pure Python implementation. A second version will use the `@jit` decorator but no parallelization. The third uses `implicit` parallelization and the fourth `explicit` parallelization.


```python
# Reference implementation without Numba optimization
def abc_model_py(params, rain):
    # initialize model variables
    outflow = np.zeros((rain.size, params.size), dtype=np.float64)

    # loop over parameter sets
    for i in range(params.size):
        # unpack model parameters
        a = params['a'][i]
        b = params['b'][i]
        c = params['c'][i]

        # Reset model states
        state_in = 0
        state_out = 0

        # Actual simulation loop
        for j in range(rain.size):
            state_out = (1 - c) * state_in + a * rain[j]
            outflow[j,i] = (1 - a - b) * rain[j] + c * state_in
            state_in = state_out
    return outflow

# Jit'ed but not parallelized implementation
@jit(nopython=True)
def abc_model_jit(params, rain):
    # initialize model variables
    outflow = np.zeros((rain.size, params.size), dtype=np.float64)

    # loop over parameter sets
    for i in range(params.size):
        # unpack model parameters
        a = params['a'][i]
        b = params['b'][i]
        c = params['c'][i]

        # Reset model states
        state_in = 0
        state_out = 0

        # Actual simulation loop
        for j in range(rain.size):
            state_out = (1 - c) * state_in + a * rain[j]
            outflow[j,i] = (1 - a - b) * rain[j] + c * state_in
            state_in = state_out
    return outflow

# Implementation with implicit parallelization
@jit(nopython=True, parallel=True)
def abc_model_impl(params, rain):
    # initialize model variables
    outflow = np.zeros((rain.size, params.size), dtype=np.float64)

    # loop over parameter sets
    for i in range(params.size):
        # unpack model parameters
        a = params['a'][i]
        b = params['b'][i]
        c = params['c'][i]

        # Reset model states
        state_in = 0
        state_out = 0

        # Actual simulation loop
        for j in range(rain.size):
            state_out = (1 - c) * state_in + a * rain[j]
            outflow[j,i] = (1 - a - b) * rain[j] + c * state_in
            state_in = state_out
    return outflow

# Implementation with explicit parallelization (see prange in 1st loop)
@jit(nopython=True, parallel=True)
def abc_model_expl(params, rain):
    # initialize model variables
    outflow = np.zeros((rain.size, params.size), dtype=np.float64)

    # loop over parameter sets
    for i in prange(params.size):
        # unpack model parameters
        a = params['a'][i]
        b = params['b'][i]
        c = params['c'][i]

        # Reset model states
        state_in = 0
        state_out = 0

        # Actual simulation loop
        for j in range(rain.size):
            state_out = (1 - c) * state_in + a * rain[j]
            outflow[j,i] = (1 - a - b) * rain[j] + c * state_in
            state_in = state_out
    return outflow
```

Okay now we gonna generate random model parameters and a random array of precipitation. Also look how we can use our numpy data type here in combination with the `astype()` function.


```python
params = np.random.random(8).astype(abc_dtype)
rain = np.random.random(10**6)
```

Now we gonna use the `timeit` module to compare the runtimes of each of the four implementations.


```python
time_py = %timeit -o abc_model_py(params, rain)
```

    7.63 s ± 59.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


```python
time_jit = %timeit -o abc_model_jit(params, rain)
```

    66.9 ms ± 2.23 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


```python
time_impl = %timeit -o abc_model_impl(params, rain)
```

    54.8 ms ± 329 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)


```python
time_expl = %timeit -o abc_model_expl(params, rain)
```

    22.3 ms ± 1.67 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


```python
time_py.best/time_expl.best
```

    402.9802136214722


Between the pure Python version and the explicit parallelized version there is roughly a 400 x time difference! And through explicit parallelization we could improve the runtime by a factor of 3, while implicit parallelization only gave us a minor speed up for this function. The numbers above come from running this code on a Intel Xeon(R) CPU E5-1620 v3 @ 3.50GHz × 8.

Since with parallel computation weired stuff can happen, let's also make sure all function outputs are identical.


```python
outflow_py = abc_model_py(params, rain)
outflow_jit = abc_model_jit(params, rain)
outflow_impl = abc_model_impl(params, rain)
outflow_expl = abc_model_expl(params, rain)
```


```python
if (np.array_equal(outflow_py, outflow_jit) and
    np.array_equal(outflow_py, outflow_impl) and
    np.array_equal(outflow_py, outflow_expl)):
    print("All output matrices are identical.")
```

    All output matrices are identical.


Okay so that's it for now. I hope this post helps you to understand how custom Numpy data types can be used in combination with the Numba library and how you can make Numba parallelize your code to gain additional speedups

You can find this entire article as Jupyter Notebook [here](https://github.com/kratzert/numba_tutorials/blob/master/Part_2_custom_dtypes_and_parallelization.ipynb).
