# vim: spell spelllang=en
import functools
import numpy
from . import pypar
from .util import balance


class MasterSlavePair(object):
    """MPI master-slave pair.

    The class MasterSlavePair implements a simple workflow using master-slave
    parallel processes.  It works for the following scenario:
        * One has a function to be applied to a one-dimension data array
          (presumably large enough to justify parallelization).  This makes the
          problem embarrassingly parallelizable: each slave just need to apply
          the function on its portion of data.
        * The function evaluation relies on another parameter, a fixed- (and
          known-) size numpy array ins, i.e. the "instruction".  Instructions
          can be generated in other parts of the program and sent to the
          evaluator for evaluation, while data remains constant.
        * The function should be called as function(ins, data), where data
          is the portion of data for each slave.  It should return a
          1-dimensional numpy array (i.e. same size as data).
        * The function should not alter the input data as a side effect.
        * Only the master should interact with the "rest of the world", i.e.
          receiving instructions and returning values.

    The class provides an eval() method for this purpose.  It accepts the
    instruction as the only input parameter, and returns the value as evaluated
    by the parallel slaves.  Normally, this is the only method that the user
    should care about and use frequently.

    The class also provide the following methods:
        * terminate():  When called, send an array of NaN's to the slaves,
          which put the slaves out of the read-evaluate-send loop, thus
          put them into an essentially "terminated" state.
        * confineto():  This is suitable for use as a function wrapper
          (and fits into the decorator syntax).  The wrapped function will be
          confined to specific rank(s), outside of which it will not be called
          (a surrogate can be provided).

    During initialization, information about MPI runtime (such as ranks) will
    be collected, and information useful for scheduling (such as slice_table, a
    collection of indices that specifies which portion of data each slave
    should work on) are computed.  They will be saved as the instance's
    attributes.  It is expected that they're not manually modified.

    There must be at least one slave to work on the data.  If not, RuntimeError
    will be raised.

    Implementation detail:
        So far, the class is implemented using the pypar library, which
        interfaces an available, underlying MPI implementation.

        Instructions are sent using MPI broadcast (MPI_Bcast).  Due to the
        limitation of pypar, there's no choice of communicator but
        MPI_COMM_WORLD.  This may change in the future.

        Each portion of the result is collected to the master using
        an individual MPI_Send/MPI_Receive.  When the message size is small,
        and number of processes is not large, this appears to have lower
        latency than MPI_Gatherv (see also http://arxiv.org/abs/1103.5616).
        Again, this may change in the future, and we may refactor the class
        so that the user can choose to use another messaging strategy, such as
        scatter/gather, by inheriting from this class.
    """
    def __init__(self, work_callback, ins_size, data,
                 master_rank=0, tag_data=0xda7a):
        """Initialize the master-slave pair.  Must be called by all
        MPI processes with the same input parameters.

        work_callback:  work function of signature
        work_callback(instruction, data).

        ins_size:  size of instruction array.

        data:  entire dataset to be worked on.

        master_rank:  rank of master process, default to 0.

        tag_data:  MPI tag for sending the evaluated results back to master.
        Must be integer.  Default to 55930 (in hex: 0xda7a).
        """
        self.size = pypar.size()
        if self.size <= 1:
            raise RuntimeError("No slaves")
        self.rank = pypar.rank()
        self.nslaves = self.size - 1
        self.ins_size = ins_size
        self.master_rank = master_rank
        self.ismaster = self.rank == master_rank
        self.work_size = data.shape[0]
        self.slice_table = [balance(self.work_size, self.nslaves, i)
                            for i in xrange(self.nslaves)]
        self.tag_data = tag_data
        if not self.ismaster:
            # Slave-specific initialization
            self.ins_buf = numpy.empty(self.ins_size)
            self.slave_started = False
            self.myslaveid = (self.rank -
                              (1 if self.rank > self.master_rank else 0))
            self.data = numpy.ascontiguousarray(data)
            self.work_callback = work_callback
            self.mylow, self.myhigh = self.slice_table[self.myslaveid]
            # Put slave-self into working state
            self.startslave()
        else:
            self.res_buf = numpy.empty(self.work_size)
            rsize = self.slice_table[0][1] - self.slice_table[0][0]
            self.masterrecvbuf = numpy.empty(rsize)

    def master(self, ins, interlude_callback=None, *i_args, **i_kwargs):
        """Run master (instructor) with input instruction array ins.

        This method returns None and alter the state of master as side effect.
        Normally not needed by the user.  See eval() instead.

        Optionally, this method also takes an "interlude" callable, along with
        its arguments.  During the intermission between the end of
        starting-slaves and collecting first results from slave, rather than
        doing nothing, the master can use this lapse of time to call the
        given callback and save its results, if any.  This result will be
        returned.

        Of course, it isn't worthwhile to call the interlude if itself hogs
        the execution time of master.
        """
        pypar.broadcast(ins, root=self.master_rank, bypass=True)
        if callable(interlude_callback):
            ires = interlude_callback(*i_args, **i_kwargs)
        else:
            ires = None
        work_done = 0
        while work_done < self.nslaves:
            n, stat = pypar.receive(source=pypar.any_source, tag=self.tag_data,
                                    buffer=self.masterrecvbuf, bypass=True,
                                    return_status=True)
            thisslave = (stat.source -
                         (1 if stat.source > self.master_rank else 0))
            sl, sh = self.slice_table[thisslave]
            self.res_buf[sl:sh] = self.masterrecvbuf[:sh - sl]
            work_done += 1
        return ires

    def eval(self, ins, interlude_callback=None, *i_args, **i_kwargs):
        """Evaluate the result of applying the function work_callback,
        as given during initialization, for the instruction ins.

        Essentially, it can be thought of as a parallel version of calling
        work_callback(ins, data).

        Returns a tuple (mainres, interlude_res), where the first element
        is numpy array that has the same value as obtained by the
        non-parallel call, when called in master.  The 2nd element is the
        return value of the interlude call.
        """
        ins = numpy.ascontiguousarray(ins)
        if self.ismaster:
            ires = self.master(ins, interlude_callback, *i_args, **i_kwargs)
            return (self.res_buf, ires)
        else:
            return (None, None)

    def slave(self):
        """Slave program.

        Normally, not needed by the user, and manually calling it could
        easily break the program.

        The slave is put into a read-evaluate-send loop, awaiting input
        instruction from master.  When the slave exits the loop (due to command
        from master), returns None.
        """
        while True:
            pypar.broadcast(self.ins_buf, root=self.master_rank, bypass=True)
            if numpy.isnan(self.ins_buf).all():  # should exit
                break
            else:                                # should work on instruction
                res = self.work_callback(self.ins_buf,
                                         self.data[self.mylow:self.myhigh])
                pypar.send(res, self.master_rank, tag=self.tag_data,
                           use_buffer=True, bypass=True)

    def startslave(self):
        """Normally not need by the user.  Automatically called during
        initialization.

        Start the slave loop, with guard against calling slave() multiple
        times.  When called from master, does nothing.  Returns None.
        """
        if self.ismaster:
            return None
        # Start slave only once
        if not self.slave_started:
            self.slave_started = True
            self.slave()

    def terminate(self):
        """Terminate the slaves by broadcasting NaN's to them as instruction.
        """
        if self.ismaster:
            ins = numpy.array([numpy.nan] * self.ins_size)
            pypar.broadcast(ins, root=self.master_rank, bypass=True)

    def confineto(self, function, rankpredicate=None, otherwise=None):
        """Function wrapper (suitable for use in decorator syntax) to make
        the wrapped function's body executed only in the given rank.

        rankpredicate is a callable that takes a rank and returns a boolean
        value.  The truth of its return value is used to determine whether
        function (if True) or otherwise (if False) should be called.
        By default, if rankpredicate is None, the rank-logic is "to confine
        to the 'master' rank".

        If otherwise is a callable, it is called instead of the wrapped
        function.

        In case no function gets actually called, returns None.
        """
        if rankpredicate is None:
            def rankpredicate(rank):
                return rank == self.master_rank

        @functools.wraps(function)
        def wrap(*args, **kwargs):
            if rankpredicate(self.rank):
                return function(*args, **kwargs)
            else:
                if callable(otherwise):
                    return otherwise(*args, **kwargs)
                else:
                    return None
        return wrap


__all__ = ["MasterSlavePair"]
