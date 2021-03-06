# vim: spell spelllang=en
import operator


def balance(ntask, nworker, idxworker):
    """Similar to pypar.balance, but when idxworker > (nworker - 1), returns
    zero-sized slice.
    """
    ntask = int(ntask)
    nworker = int(nworker)
    idxworker = int(idxworker)
    if idxworker >= nworker:     # If idxworker "index out of range"
        return (ntask, ntask)
    divisor, remainder = ntask // nworker, ntask % nworker
    # First remainder workers receive an additional task
    if idxworker < remainder:
        low = idxworker * (divisor + 1)
        high = low + divisor + 1
    else:
        low = remainder + idxworker * divisor
        high = low + divisor
    return (low, high)


def balance_gatherv(ntask, nworker):
    """Similar to balance(), but returns (counts, offsets) suitable
    for MPI_Gatherv().
    """
    ntask = int(ntask)
    nworker = int(nworker)
    divisor, remainder = ntask // nworker, ntask % nworker
    counts = [divisor + (1 if i < remainder else 0) for i in xrange(nworker)]
    offsets = list(scanl(counts))[:-1]
    return (counts, offsets)


def balance_gatherv_skipmanager(ntask, nworker, managerrank=0):
    """Similar to balance_gatherv(), but make sure that the manager contributes
    only a zero-sized section.  The 2nd argument is interpreted as the
    number of *workers*, which is usually (nprocess - 1).  By using "workers",
    it is implied that the manager doesn't do worker work.
    """
    rawcounts, rawoffsets = balance_gatherv(ntask, nworker)
    counts = rawcounts[:managerrank] + [0] + rawcounts[managerrank:]
    offsets = (rawoffsets[:managerrank] + [rawoffsets[managerrank]] +
               rawoffsets[managerrank:])
    return (counts, offsets)


def scanl(iterable, f=operator.add, starting=0):
    """Generator similar to Haskell's scanl."""
    it = iter(iterable)
    value = starting
    yield value
    for x in it:
        value = f(value, x)
        yield value


__all__ = ["balance", "balance_gatherv", "balance_gatherv_skipmanager",
           "scanl"]
