# vim: spell spelllang=en
# Just import pypar, so that MPI_Init() is called (exactly once), and then
# "export" MasterSlavePair.  The util module doesn't need special attention.
"""parallelvicd:  a package for one kind of simple MPI master-slave workflow.

See the documentation to the MasterSlavePair class for details.

The package also provides a utility module util with some useful functions.
"""
import pypar
from .drive import MasterSlavePair
