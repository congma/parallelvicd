<!--
vim: spell spelllang=en
-->
parallelvicd:  Python package for one kind of simple MPI manager-worker
workflow

---

Summary
-------

This package `parallelvicd` provides a simple manager-worker class for
applying a user-defined function to an array of inputs.  It is written using
the [`pypar`](https://github.com/daleroberts/pypar "pypar project repository")
package in Python.

To quote the docstring of `ManagerWorkerPair` class, it is specifically written
to work on the following problem:

 * One has a function to be applied to a one-dimension data array
   (presumably large enough to justify parallelization).  This makes the
   problem embarrassingly parallelizable: each worker just need to apply
   the function on its portion of data.
 * The function evaluation relies on another parameter, a fixed- (and
   known-) size `numpy` array `ins`, i.e. the "instruction".  Instructions
   can be generated in other parts of the program and sent to the
   evaluator for evaluation, while data remains constant.
 * The function should be called as `function(ins, data)`, where data
   is the portion of data for each worker.  It should return a
   1-dimensional `numpy` array (i.e. same size as data).
 * The function should not alter the input data as a side effect.
 * Only the master should interact with the "rest of the world", i.e.
   receiving instructions and returning values.

For a detailed documentation, please read the docstring of the
`ManagerWorkerPair` class.

Example usage
-------------

```python
import numpy
import parallelvicd


numpy.set_printoptions(precision=2)	# Avoid too many fractional digits.


def worker_routine(instruction, data):  # Our example worker program.
    addend = instruction.max()
    return data + addend


data = numpy.arange(12.0)		# Fake some data.
# By creating the manager-worker pair like this, a pool of workers is put into
# a ready state.
# Note that we have fixed the size of instruction array as 3.
pair = parallelvicd.ManagerWorkerPair(worker_routine, 3, data)
# MARK: worker now entered the evaluation loop and won't go beyond this mark
# unless told to terminate the loop.


# Example of a function that is confined to the manager process.
# In this case, it's a simple I/O function that prints something to stdout.
@pair.confineto
def print_result(array):
    print array


# Example of using the parallel evaluator.  Now, the name `res` will have value
# as evaluated by the workers in the manager process, but the worker processes
# are trapped in the evaluation loop and can't even reach here.
res = pair.eval(numpy.array([0.1, 0.0, -0.1]))[0]

# Try to use the result.  Note that we pass it to a function confined to
# manager, so that even after the worker break out of the evaluation loop
# eventially (and hence get None as value of `res`), we don't get extra output.
print_result(res)			# Should print the correct result.

# pair.eval() can be called again
print_result(pair.eval(numpy.array([-1.0, 0.0, 0.2]))[0])

# Example of cleaning up after we're done using the pair.
# Note that the worker program, before we call terminate(), has been trapped
# in the evaluation loop.  After we call this, it will break out of the loop
# and execute whatever code from the place we noted by a comment "MARK".
# However, actually it won't do much, because eval() does nothing but
# returning (None, None) to the worker, and this is also the case of
# terminate().  In addition, we have already made print_result() confined to
# manager, so actually the worker will produce no side-effect.
pair.terminate()

# After MPI finalization, it is no longer possible to call any MPI function.
parallelvicd.finalize()			# Calls underlying MPI_Finalize().
```

To run the example, save it as `example.py` and use the `mpiexec` command to
start it (at least one worker is required):

```console
$ mpiexec -n 2 python example.py
[  0.1   1.1   2.1   3.1   4.1   5.1   6.1   7.1   8.1   9.1  10.1  11.1]
[  0.2   1.2   2.2   3.2   4.2   5.2   6.2   7.2   8.2   9.2  10.2  11.2]
```

Author
------

Cong Ma (c) 2015

License
-------

GNU GPLv3 (in compliance with the licensing terms of `pypar`).

See the file `COPYING`.
