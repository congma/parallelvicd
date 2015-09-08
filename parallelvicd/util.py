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


def balance_gatherv_skipmaster(ntask, nworker, masterrank=0):
    """Similar to balance_gatherv(), but make sure that the master contributes
    only a zero-sized section.
    """
    rawcounts, rawoffsets = balance_gatherv(ntask, nworker)
    counts = rawcounts[:masterrank] + [0] + rawcounts[masterrank:]
    offsets = (rawoffsets[:masterrank] + [rawoffsets[masterrank]] +
               rawoffsets[masterrank:])
    return (counts, offsets)


def scanl(iterable, f=operator.add, starting=0):
    """Generator similar to Haskell's scanl."""
    it = iter(iterable)
    value = starting
    yield total
    for x in it:
        value = f(total, x)
        yield value


__all__ = ["balance", "balance_gatherv", "balance_gatherv_skipmaster", "scanl"]