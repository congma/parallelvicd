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
