"""Module for MPI-based communication.

Implements a server/client based model via class ServerClient. One
server (root) dispatches the work to N-1 workers (ranks). The workers
send signals to root, each having a different meaning:

  signal   meaning                              root's response
  ------------------------------------------------------------------------
  1        worker request more work             sends more work (if any work left)
  2        worker done, wants to send results   receive results from worker

See class doc strings for more information.

"""

__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'
__version__ = '2014-12-02'
__version_first__ = '2011-05-03'


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
class Communicator:
    """Class for MPI signaling with server/clients model.

    Server is called 'root' and has MPI index 0. Clients are called
    'ranks' and have MPI indices 1...n

    This class provides several communication methods. Some send and
    receive serialized data, some rely on a writable memory buffer
    object. For the latter, the class instantiates a 'signalbuffer'
    object. This has to be be a Python object that exports a writable
    buffer interface, like objects provided by the built-in array
    module for instance. 'signalbuffer' is a 2-element array of
    integers, holding [rank of sender,signal]. 'rank' is a value from
    1...Communicator.size, indicating the sending rank. 'signal' can
    be either 1 or 2 (1 indicating a work request, 2 announcing the
    intent to send follow-up data).

    See docstrings methods for more information.

    """

    # imports
    from array import array
    from mpi4py import MPI

    def __init__(self):
      
      self.comm = Communicator.MPI.COMM_WORLD
      self.root = 0
      self.rank = self.comm.rank   # alias
      self.size = self.comm.size   # alias
      self.signalbuffer = Communicator.array('i',[0,0]) # 'i' for signed integer
      self.databuffer = None # _must_ be replaced with a numpy array (float) externally before calling this .Send() method


    def Barrier(self):
      """Synchronize all ranks in the communicator."""

      self.comm.Barrier()


    def bcast(self,message):
      """Broadcast a message from root to all ranks.

      Here, all broadcasted messages originate in root, thus the source
      'root' can be hard-wired.
      """

      return self.comm.bcast(message,0)


    def send(self,message,receiver=0):
      """Send message to receiver.

      Default receiver is root.
      """

      self.comm.send(message,receiver)


    def recv(self,sender=0):
      """Receive a message from sender.

      Default sender is root. Return the received message.
      """

      return self.comm.recv(source=sender)


    def Send(self,receiver=0):
      """Send 'databuffer' (a numpy array) to receiver (via writable buffer interface).

      Default receiver is root (=0).
      """

      self.comm.Send([self.databuffer,Communicator.MPI.FLOAT],receiver)


    def Recv(self,source=None):
      """Receive 'databuffer' (a numpy array) from source (via writable buffer interface).
      """

      self.comm.Recv([self.databuffer, Communicator.MPI.FLOAT], source=source)  # BUGFIX


    def sendsignal(self,signal,receiver=0):
      """Send a signal to receiver via objects with a writable buffer interface.

      Signal is an integer, and either 1 or 2. Default receiver is root
      (=0). The writable buffer-exposing interface is provided by the
      Python standard module 'array'.
      """

      self.signalbuffer[0] = self.rank   # set 1st element of buffer to rank
      self.signalbuffer[1] = signal      # set 2nd element of buffer to signal
      self.comm.Send([self.signalbuffer,Communicator.MPI.INT],receiver)


    def receivesignal(self):
      """Receive a signal from any sender via objects with a writable buffer interface.

      Signal is an integer, and either 1 or 2. The writable
      buffer-exposing interface is provided by the Python standard
      module 'array'. Don't proceed until a signal has been
      received. Return (sender,signal).
      """

      req = self.comm.Irecv([self.signalbuffer,Communicator.MPI.INT],source=Communicator.MPI.ANY_SOURCE)   # listen for a message from any rank
      req.Wait()                           # don't move any further until received such message
      sender, signal = self.signalbuffer   # received 'signalbuffer' is a 1-dim integer array of length 2; holds (rank of sender,signal)
      return sender, signal


    def mpifinalize(self):
      """Finalize MPI run."""

      Communicator.MPI.Finalize()
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
class ServerClient(Communicator):
    """Perform a task in parallel via a server-client model.

    Description
    -----------
    This class instatiates first an MPI communicator object for all
    ranks (including root). It then establishes the necessary
    communication between root and all other ranks (workers/clients) of
    the communicator. The communication is a server-client based, with
    the MPI-root being the server.

    All workers signal the server with requests for work, then perform
    the task independently, using a permutation of parameters supplied
    in a list. An integer index is also supplied by the server; it is
    the next available from range(problemsize), 'problemsize' being the
    number of times a job has to be performed in total (with different
    parameter values each time probably).

    When done with the task, a worker signals the root again, announcing
    that it wishes to return a result (typically a message, or a list of
    values, etc.) Upon such signal root receives the result, and the
    worker is free to receive more work from root.

    If the list of work is exhausted, root will send 'BREAK' to a worker
    who is requesting more work. The worker will then stop requesting
    more work within this task.

    The task performed by a worker is in the function 'workfunc'. A
    reference to such a user-written function is to be provided when
    executing the task. 'Executing the task' means calling the __call__
    method of this class.

    Root can collect the results being returned by the workers. In that
    case, a reference to a 'result function' is to be provided upon
    invoking the __call__ method. The result function will be executed
    by the server every time it has received a new result from a
    worker. In a typical scenario the result function might be a
    collecting function, such as list concatenation, or directory
    updating. If there is no need for a result function, simply pass
    None when calling the agent.

    The server and clients instantiated via this class can be re-used
    for different tasks. When no longer needed, at the end of the
    program, simply terminate the MPI communicator via
    agent.mpifinalize()

    'generatorfunc' generates the command to be send to the requesting
    worker.

    Examples
    --------
    Intialize:

      # Initialize the ServerClient object.  This will instantiate a
      # server for root, and clients for all other ranks.  Internally
      # this establishes an MPI communicator object.
      agent = ServerClient()

    Run a task in parallel:

      # Define some worker function (only clients execute this). Inside,
      # you can use idx, or value, or both, or none, whatever gets the
      # job done.
      def myworker(idx,value,**kwargs):     
          return 0.5*value

      # Define some result function (only server executes this).
      def myresult(idx,li,a,**kwargs):
          li.append(a)
          return li

      # in-line generator (can also be a full fledged generator function instead).
      mygen = (j for j in range(8))

      # this runs a task on all agents
      agent(mygen,myworker,myresult)
      print agent.results
      >>> [0.5, 0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]   # this is not necessarily sorted!

    Both the worker function and the result function accept (the same)
    optional, user-suppliable arguments (in this example one called 'const'):

      # look for an optional argument passed in via **kwargs
      def myworker(idx,value,**kwargs):
          aux = kwargs['const']
          return 0.5*value - aux

      # run a task, supplying an optinal argument to the worker func
      agent(mygen,myworker,myresult,const=0.5)
      print agent.results
      >>> [0.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]   # this is not necessarily sorted!

    """

    def __init__(self):
      """Instantiate a ServerClient class instance."""
      
      Communicator.__init__(self)   # subclassing the Communicator class


    def __call__(self,generatorfunc,workfunc,resultfunc,**kwargs):
      """Calling the instance of this class.

      If rank == root (aka the server), runs generatorfunc, and passes
      the result of this to the next rank/worker asking for work.

      If rank != root (aka worker), receives the next chunk of work (aka
      next permutation of parameters) and runs the worker function. The
      worker function may or may not return something upon
      completion. This returned value(s) will be handled by root, and
      there via function 'resultfunc'. The result function is
      optional. If you don't need one, specify the value None instead
      when calling this class instance."""

      # --- ROOT / SERVER ---
      if self.rank == 0:
        self.results = None
        self.rfunc = resultfunc  # resultfunc is a function handle, provided from outside
        self.nbreaks = 0
        self.generator = generatorfunc

        idx = 1  # counter
        while True:  # loop es long as work is being requested by the ranks
          if self.nbreaks == self.size-1:  # break out of while loop once every rank has terminated
            break

          sender, signal = self.receivesignal()
          if signal == 1:  # a rank requests work
            try:
              command = self.generator.next()
              self.send((idx,command,kwargs),sender)
            except StopIteration:
              self.send((-1,'BREAK',kwargs),sender)
              self.nbreaks +=1
              continue

            idx += 1

          elif signal == 2: # this means sender is done, wants to rend results
            (origidx,res) = self.recv(sender=sender)
            if self.rfunc:   # only call result function of not None was passed
              self.results = self.rfunc(origidx,self.results,res,**kwargs)

      # --- WORKER RANKS / CLIENTS ---
      else:

        self.wfunc = workfunc  # workfunc is a function handle, provided from outside

        while True:
          self.sendsignal(1)                       # tell root that we want work

          (idx,command,kwargs) = self.recv()   # receive job instructions from root
          if command == 'BREAK':
            break

          # now do our own work by calling a worker function with the index idx as argument
          res = self.wfunc(idx,command,**kwargs)

          # once done, send signal(2) to root (= we're done here, want to send results)
          self.sendsignal(2)
          self.send((idx,res))

      # --- SYNCHRONIZE ALL RANKS ---
      self.Barrier()
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
