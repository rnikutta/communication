"""Minimal example to demonstrate the use of the communication.py module.

This example demonstrates how to write a work generator, worker
function(s), the result function, etc. It also shows that instantiated
workers can be re-used for other tasks.

"""

__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'
__version__ = '2014-12-03'


def workerfunc1(idx,value,**kwargs):
    """Worker function.

    Get input, process it, return.

    idx : int
        Integer unique identified in current collaborative
        task. Assigned by the Server. Useful for instance if keeping
        track of things inside the worker function is required.

    value : anything really
        This is the Server-sent info necessary for this worker to
        perform its job. It can be a number, a string, a file path, a
        list, an array, and many more things. The worker func can
        process this value, together with idx.

    kwargs : anything
        Additional arguments can be passed from Server to this Worker
        using keyword arguments. For instance:

          ...
          a = kwargs['coefficient']
          b = kwargs['exponent']
          return a*value**b
    """

    if 'args' in kwargs.keys() and kwargs['args'].verbose:
        tagprint("Working on task ID %d" % idx)

    return idx**2  # square the input and return; we're not using value or kwargs here


def workerfunc2(idx,value,**kwargs):
    """Worker function. Sam principle as workerfunc1."""

    if 'args' in kwargs.keys() and kwargs['args'].verbose:
        tagprint("Working on task ID %d" % idx)

    return idx/5.  # divide idx by 5 and return

  
def resfunc(idx,li,a,**kwargs):
    """Result function, should be only called by root.

    idx : int
        Unique ID of the task that was calculated by a worker.

    li : anything
        Partial result being sent back from worker to root. Here it's
        a list, but this can be really almost anything (as long as
        it's pickleble).

    kwargs : anything
       Optional additional keyword arguments.

    """

    # on first call make an empty list to collect the results from the workers
    if not li:
        li = []  

    # 'process' the partial result; here we simply append them to list
    li.append(a)

    return li


def gen(n):
    """A very simple generator of work.

    Produces input parameters for the workers. Here it's just a stream
    of integers.

    n : int
       How many separate 'jobs' to run by all the workers?

    """

    for j in xrange(n):
        yield j
  

def get_args():
    """MPI-friendly parser for command-line args.

    This func is called by all ranks, but inside it differentiates
    between root and other ranks. The purpose is to be able to exit
    all ranks cleanly if a problem occurs with command line args, or
    if the -h (help) flag is given.
    """

    args, msg = None, None
    if agent.rank == 0:
        import argparse
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description=__doc__,\
                                         usage='\n    mpirun -np N python %s [--help] [--verbose]\n\nwhere N (>1) is the total number of processes (1 root + N-1 workers).' % sys.argv[0])
        parser.add_argument('-v','--verbose',help="Verbose output on. Skip this chatty option if preferred.",action="store_true")

        try:
            args = parser.parse_args()
        except IOError, msg:
            print msg
        except SystemExit:
            pass

    msg = agent.bcast(msg)  # broadcast error message from parsing, if any
    if msg:
        sys.exit()

    args = agent.bcast(args)   # broadcast command line options, after parsing

    if not args:               # if args == None, -h or -V flag was given; exit all ranks gracefully
        sys.exit()

    return args


def tagprint(msg,*args):
    """Pretty-print a message, tagging it with the agent's ID."""
  
    if not args: args = ''
    print "[rank %03d] %s" % (agent.rank, msg), args


def finalize():
    # finalize MPI (everyone)
    agent.Barrier()   # wait for all workers to finish
    tagprint("Finalizing.")
    agent.mpifinalize()
    assert agent.MPI.Is_finalized()


# \\\ MAIN PROGRAM
if __name__ == '__main__':

    import sys
    import warnings
    from communication import ServerClient

    # Instantiate server/client objects with an MPI communicator
    # built-in. MPI root is server, other ranks are clients /
    # workers. All are some form of 'agents', so let's call them that.
    agent = ServerClient()

    if agent.size == 1:
        warnings.warn("No worker ranks found. The number of processes must be at least 2. Nothing will be computed.")

    # check command-line args
    args = get_args()

    # task 1
    g1 = gen(17)  # work generator (here a very simple generator of integers)
    agent(g1,workerfunc1,resfunc,args=args)  # signature: (generator,workerfunc,resultfunc,**kwargs)
    agent.Barrier()   # wait for all workers to finish
    if agent.rank == 0:
        tagprint("Results task 1: ",agent.results)

    # task 2: the pool of agents/workers can be re-used for another task
    g2 = gen(23)
    agent(g2,workerfunc2,resfunc,args=args)  # signature: (generator,workerfunc,resultfunc,**kwargs)
    agent.Barrier()
    if agent.rank == 0:
        tagprint("Results task 2: ",agent.results)

    # finalize MPI (everyone)
    finalize()   # has a Barrier() built-in
