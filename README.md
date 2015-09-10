<!--
vim: spell spelllang=en
-->
parallelvicd:  Python package for one kind of simple MPI master-slave workflow

---

Summary
-------

This package `parallelvicd` provides a simple master-slave class for
applying a user-defined function to an array of inputs.  It is written using
the [`pypar`](https://github.com/daleroberts/pypar "pypar project repository")
package in Python.

To quote the docstring of `MasterSlavePair` class, it is specifically written
to work on the following problem:

 * One has a function to be applied to a one-dimension data array
   (presumably large enough to justify parallelization).  This makes the
   problem embarrassingly parallelizable: each slave just need to apply
   the function on its portion of data.
 * The function evaluation relies on another parameter, a fixed- (and
   known-) size `numpy` array `ins`, i.e. the "instruction".  Instructions
   can be generated in other parts of the program and sent to the
   evaluator for evaluation, while data remains constant.
 * The function should be called as `function(ins, data)`, where data
   is the portion of data for each slave.  It should return a
   1-dimensional `numpy` array (i.e. same size as data).
 * The function should not alter the input data as a side effect.
 * Only the master should interact with the "rest of the world", i.e.
   receiving instructions and returning values.

For a detailed documentation, please read the docstring of the
`MasterSlavePair` class.

Example usage
-------------

```python
import numpy
import parallelvicd


numpy.set_printoptions(precision=2)	# Avoid too many fractional digits.


def worker(instruction, data):		# Our example worker program.
    addend = instruction.max()
    return data + addend


data = numpy.arange(12.0)		# Fake some data.
# By creating the master-slave pair like this, a pool of slaves is put into a
# ready state.
# Note that we have fixed the size of instruction array as 3.
pair = parallelvicd.MasterSlavePair(worker, 3, data)
# MARK: slave now entered the evaluation loop and won't go beyond this mark
# unless told to terminate the loop.


# Example of a function that is confined to the master process.
# In this case, it's a simple I/O function that prints something to stdout.
@pair.confineto
def print_result(array):
    print array


# Example of using the parallel evaluator.  Now, the name `res` will have value
# as evaluated by the slaves in the master process, but the slave processes
# are trapped in the evaluation loop and can't even reach here.
res = pair.eval(numpy.array([0.1, 0.0, -0.1]))[0]

# Try to use the result.  Note that we pass it to a function confined to
# master, so that even after the slaves break out of the evaluation loop
# eventially (and hence get None as value of `res`), we don't get extra output.
print_result(res)			# Should print the correct result.

# pair.eval() can be called again
print_result(pair.eval(numpy.array([-1.0, 0.0, 0.2]))[0])

# Example of cleaning up after we're done using the pair.
# Note that the slave program, before we call terminate(), has been trapped
# in the evaluation loop.  After we call this, it will break out of the loop
# and execute whatever code from the place we noted by a comment "MARK".
# However, actually it won't do much, because eval() does nothing but
# returning (None, None) to the slave, and this is also the case of
# terminate().  In addition, we have already made print_result() confined to
# master, so actually the slave will produce no side-effect.
pair.terminate()

# After MPI finalization, it is no longer possible to call any MPI function.
parallelvicd.finalize()			# Calls underlying MPI_Finalize().
```

To run the example, save it as `example.py` and use the `mpiexec` command to
start it (at least one slave is required):

```console
$ mpiexec -n 4 python example.py
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
