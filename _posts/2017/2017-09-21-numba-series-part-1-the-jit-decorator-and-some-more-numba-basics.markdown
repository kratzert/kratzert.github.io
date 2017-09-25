---
layout: post
title: 'Numba series part 1: The @jit decorator and some more Numba basics'
date: '2017-09-21 21:41'
excerpt: >-
  In this part we'll have a closer look at the @jit decorator of the Numba
  library and talk about some pitfalls, as well as some more basics.
comments: true
---

In the first part of the little Numba series I've planned we will focus mainly on the `@jit` decorator. Their exist different decorators in the Numba library and we will talk about them later, but for the start we will concentrate on the `@jit` one. On our way we will also explore some basics, which are good to know about Numba library in general.

Now to the decorator: The `@jit` decorator is maybe the central feature of the Numba library and stands, as you might guess, for just-in-time compilation. It's also possible to let Numba compile code ahead-of-time (aot), which we will discuss in another article.

As in the [introduction article](https://kratzert.github.io/2017/09/12/introduction-to-the-numba-library.html), we will use the simple hydrological model again, as the function we want to speed up with Numba. With this model to optimize, we will explore the different options you have, when using the `@jit` decorator. I also want to highlight the [official documentation](http://numba.pydata.org/numba-doc/0.35.0/), in this case especially for the [@jit decorator](http://numba.pydata.org/numba-doc/0.35.0/user/jit.html), since I think they are really well written (and make this post maybe unnecessary).

So let's get started.


```python
import timeit
import numpy as np

from numba import jit
```

As already seen in the previous article, the `@jit` decorator can be used without any arguments. Just adding the decorator to your function tells Numba that this function should be compiled. Let's have a look at two different ways, how we could implement the hydrological ABC-Model. Focus on the inputs.


```python
@jit
def abc_model_1(a, b, c, rain):
    """First implementation of the ABC-Model.

    Args:
        a, b, c: Model parameter as scalars.
        rain: Array of input rain.

    Returns:
        outflow: Simulated stream flow.

    """
    # Initialize model variables
    outflow = np.zeros((rain.size), dtype=np.float64)
    state_in = 0
    state_out = 0

    # Actual simulation loop
    for i in range(rain.size):
        state_out = (1 - c) * state_in + a * rain[i]
        outflow[i] = (1 - a - b) * rain[i] + c * state_in
        state_in = state_out
    return outflow


@jit
def abc_model_2(params, rain):
    """Second implementation of the ABC-Model.

    Args:
        params: A dictionary, containing the three model parameters.
        rain: Array of input rain.

    Returns:
        outflow: Simulated stream flow.

    """
    # Initialize model variables
    outflow = np.zeros((rain.size), dtype=np.float64)
    state_in = 0
    state_out = 0

    # Actual simulation loop
    for i in range(rain.size):
        state_out = (1 - params['c']) * state_in + params['a'] * rain[i]
        outflow[i] = ((1 - params['a'] - params['b']) * rain[i]
                      + params['c'] * state_in)
        state_in = state_out
    return outflow

```

This cell can be executed without any error, so let's have a look at the runtime of this two functions. We use an array of random numbers as rain, since this isn't important here.


```python
rain = np.random.rand((10**6))
time_model_1 = %timeit -o abc_model_1(0.6, 0.1, 0.3, rain)
time_model_2 = %timeit -o abc_model_2({'a': 0.6, 'b': 0.1, 'c': 0.3}, rain)
```

    2.69 ms ± 69.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    686 ms ± 994 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)


What happened here? There was no error, but the code of the second implementation is way slower then the code of the first model implementation. So here comes the first thing you should know about Numba:

**If you use the `@jit` decorator without any arguments, Numba won't throw an error if it can't optimize your code.**

What made the problem? Well Numba does not support Python dictonaries and though it wasn't possible to optimize the second model. But instead of raising an error, it will simply fall back to normal Python. So you might think you have optimized your code, but you don't and you won't see it, unless you test for it. But their are ways how we can detect it. The first is, we could inspect the data types of the compiled function. This can be done by calling `inspect_types()` on any Numba optimized function. Here is a part of the output for the second model you would see:


```python
abc_model_2.inspect_types()
```

    abc_model_2 (pyobject, array(float64, 1d, C))
    --------------------------------------------------------------------------------
    # File: <ipython-input-2-341b0b95e9f2>
    # --- LINE 26 ---
    # label 0
    #   del $0.1
    #   del $0.5
    #   del $0.6
    #   del $0.4
    #   del $0.2
    #   del $0.8
    #   del $const0.9
    #   del state_out
    #   del $const0.10

    @jit

    # --- LINE 27 ---

    def abc_model_2(params, rain):

        # --- LINE 38 ---

        # Initialize model variables

        # --- LINE 39 ---
        #   params = arg(0, name=params)  :: pyobject
        #   rain = arg(1, name=rain)  :: pyobject
        #   $0.1 = global(np: <module 'numpy' from '/home/frederik/miniconda3/envs/numbadev/lib/python3.6/site-packages/numpy/__init__.py'>)  :: pyobject
        #   $0.2 = getattr(value=$0.1, attr=zeros)  :: pyobject
        #   $0.4 = getattr(value=rain, attr=size)  :: pyobject
        #   $0.5 = global(np: <module 'numpy' from '/home/frederik/miniconda3/envs/numbadev/lib/python3.6/site-packages/numpy/__init__.py'>)  :: pyobject
        #   $0.6 = getattr(value=$0.5, attr=float64)  :: pyobject
        #   $0.8 = call $0.2($0.4, func=$0.2, args=[Var($0.4, <ipython-input-2-341b0b95e9f2> (39))], kws=[('dtype', Var($0.6, <ipython-input-2-341b0b95e9f2> (39)))], vararg=None)  :: pyobject
        #   outflow = $0.8  :: pyobject

        outflow = np.zeros((rain.size), dtype=np.float64)

        # --- LINE 40 ---
        #   $const0.9 = const(int, 0)  :: pyobject
        #   state_in = $const0.9  :: pyobject

        state_in = 0

        # --- LINE 41 ---
        #   $const0.10 = const(int, 0)  :: pyobject
        #   state_out = $const0.10  :: pyobject
        #   jump 26
        # label 26

        state_out = 0


Whenever you see `pyobject` somewhere, it's bad. This means that Numba wasn't able to translate this variable to any data type it understands and is able to optimize. We can already see it in the function definition, as well many times below in the code.

A second way you could test your function is by using the `nopython` argument in the decorator. By setting the`nopython` to True, we tell Numba to throw an error, whenever it is not able to optimize the code. (We could also use the `@njit` decorator without any arguments, which can be seen as a short-handle to `@jit(nopython=True)`. Here is what happens when you try to compile the code of the second function with `nopython=True`.


```python
@jit(nopython=True)
def abc_model_3(params, rain):
    """Second implementation of the ABC-Model.

    Args:
        params: A dictionary, containing the three model parameters.
        rain: Array of input rain.

    Returns:
        outflow: Simulated stream flow.

    """
    # Initialize model variables
    outflow = np.zeros((rain.size), dtype=np.float64)
    state_in = 0
    state_out = 0

    # Actual simulation loop
    for i in range(rain.size):
        state_out = (1 - params['c']) * state_in + params['a'] * rain[i]
        outflow[i] = ((1 - params['a'] - params['b']) * rain[i]
                      + params['c'] * state_in)
        state_in = state_out
    return outflow
```

```python
# let's try to call this function
time_model_3 = %timeit -o abc_model_3({'a': 0.6, 'b': 0.1, 'c': 0.3}, rain)
```


    ---------------------------------------------------------------------------

    TypingError                               Traceback (most recent call last)

    <ipython-input-6-a377b3480c19> in <module>()
          1 # let's try to call this function
    ----> 2 time_model_3 = get_ipython().magic("timeit -o abc_model_3({'a': 0.6, 'b': 0.1, 'c': 0.3}, rain)")

    .
    .
    .

    File "<ipython-input-5-fe8398310597>", line 14

    This error may have been caused by the following argument(s):
    - argument 0: cannot determine Numba type of <class 'dict'>



This raises an error (I shortened the message) and as we can see at the bottom, it also tell us it wasn't able to translate the class 'dict'. The errors are not always so informative, but here it says enough.

The next thing we will have a look at are `signatures`. Until now, we didn't specify any data type of neither the input nor the output variables. If you know C/C++ or Fortran, you know that in these languages, this is a mandatory step. We can also do this with Numba. Until now, Numba checked each time we called an optimized function the data type of the input variables, looked if it already had a compiled version for these data types and if not, compiled a new version (if yes, take the already compiled version). This might be useful, if you want to make sure, only one specific data type is allowed.
Signatures are passed as string or list of strings and [here](http://numba.pydata.org/numba-doc/0.35.0/reference/types.html#numbers) you can find a list of allowed data types. If you want to pass an array instead of a scalar you will have to add `[:]` behind the data type. You also have to specify the number of dimensions of an array. This follows the semantic you know e.g. from Numpy, where `[:]` is an 1d-array, `[:,:]` a 2d-array and so one. Here is the decorator for the ABC-Model, that expects all inputs as double precision floating point numbers. You don't have to specify the data type of the output, which comes before the brackets and the input data types. If you don't, Numba tries to infer which is the data type of the output. Here is an example of a complete signature (input and output variables).


```python
@jit('float64[:](float64, float64, float64, float64[:])')
def abc_model_4(a, b, c, rain):
    """First implementation of the ABC-Model.

    Args:
        a, b, c: Model parameter as scalars.
        rain: Array of input rain.

    Returns:
        outflow: Simulated stream flow.

    """
    # Initialize model variables
    outflow = np.zeros((rain.size), dtype=np.float64)
    state_in = 0
    state_out = 0

    # Actual simulation loop
    for i in range(rain.size):
        state_out = (1 - c) * state_in + a * rain[i]
        outflow[i] = (1 - a - b) * rain[i] + c * state_in
        state_in = state_out
    return outflow

```

If we now try to call this function e.g. with a rain vector of single precision floating point values or one of the model parameter as integers, we'll get an error.


```python
rain2 = np.random.rand(10**6).astype(np.float32)
outflow = abc_model_4(0, 0.2, 0.4, rain2)
```
    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-8-17d9b0bfa8f1> in <module>()
          1 rain2 = np.random.rand(10**6).astype(np.float32)
    ----> 2 outflow = abc_model_4(0, 0.2, 0.4, rain2)


    ~/miniconda3/envs/numbadev/lib/python3.6/site-packages/numba/dispatcher.py in _explain_matching_error(self, *args, **kws)
        397         msg = ("No matching definition for argument type(s) %s"
        398                % ', '.join(map(str, args)))
    --> 399         raise TypeError(msg)
        400
        401     def _search_new_conversions(self, *args, **kws):


    TypeError: No matching definition for argument type(s) int64, float64, float64, array(float32, 1d, C)


If we try the same with the first implementation, we don't get any error. You can call `.signatures` on Numba optimized function to get a list of the signatures this function is already compiled for.


```python
# let's see if we get an error for the first function without signatures
outflow = abc_model_1(0, 0.2, 0.4, rain2)

# now let's have a look at the signatures this function already knows
abc_model_1.signatures
```




    [(float64, float64, float64, array(float64, 1d, C)),
     (int64, float64, float64, array(float32, 1d, C))]



We can see the two different signatures, the first for the first call of the beginning, and the second of this last call.

Another import point is which functions you can use within a function you would like to optimize. For the moment, these can be only very specific functions of [Pythons standard libraris](http://numba.pydata.org/numba-doc/0.35.0/reference/pysupported.html), or [Numpy functions](http://numba.pydata.org/numba-doc/0.35.0/reference/numpysupported.html) (if you click any of the links, you can see the list of supported functions and functionalities in the official Numba documentation).
For any of your own functions the rule is: You can only call functions, that are also compiled by Numba (by adding e.g. the @jit decorator).

It might not make much sense, but we could split the ABC-Model into three functions, where one calculates the new state of the storage, one calculates the outflow of the current time step and one bundles all together. This would give us something like this (I'll add the `nopython` flag, to make sure we only use Numba optimized functions and add decorators for the sake of showing how they work).

```python
@jit('float64(float64, float64, float64, float64)', nopython=True)
def get_new_state(old_state, a, c, rain):
    return (1 - c) * old_state + a * rain


@jit('float64(float64, float64, float64, float64, float64)', nopython=True)
def get_outflow(a, b, c, rain, state):
    return (1 - a - b) * rain + c * state


@jit('float64[:](float64, float64, float64, float64[:])', nopython=True)
def abc_model_5(a, b, c, rain):
    """First implementation of the ABC-Model.

    Args:
        a, b, c: Model parameter as scalars.
        rain: Array of input rain.

    Returns:
        outflow: Simulated stream flow.

    """
    # Initialize model variables
    outflow = np.zeros((rain.size), dtype=np.float64)
    state_in = 0
    state_out = 0

    # Actual simulation loop
    for i in range(rain.size):
        state_out = get_new_state(state_in, a, c, rain[i])
        outflow[i] = get_outflow(a, b, c, rain[i], state_out)
        state_in = state_out
    return outflow

```


```python
outflow = abc_model_5(0.6, 0.1, 0.3, rain)
```

Well as you see, this runs without any problems. Although this might not be the best example, I think it should be clear how this works.

There is a list of some more arguments you could pass to the `@jit` decorator ([here](http://numba.pydata.org/numba-doc/0.35.0/reference/jit-compilation.html) you can get a full list of the arguments). We'll have a look to some of them in one of the future parts of this series.

For now I'll stop here and hope you have now a better understanding how you can use the `@jit` decorator. More specific I hope you have understood, that you can only use specific functions and features in functions you want to optimize and how you can detect if your function was optimized or not. Also you should have an idea now what signatures are and how you can force specific data types with them. The last thing I told you for now was, that you can only call other functions of your own if you also compiled them using the `@jit` decorator.

In the next part, we will have a look at how we can use custom/different data types as inputs with Numba.

You can find this entire article as Jupyter Notebook [here](https://github.com/kratzert/numba_tutorials/blob/master/Part_1_the_jit_decorator.ipynb).
