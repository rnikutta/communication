communication
=============

Implements ServerClient class, which maps the server-client
communication model onto MPI. Perfect for running embarrassingly
parallel jobs on supercomputers with many nodes. Implements load
balancing: clients request more work from server and also signal when
they wish to return their results. Server distributes the work list
based on client availability, until empty. Instantiated clients are
reusable, i.e. other tasks can be worked on collaboratively after
completion of a task. This showcase includes a minimal.py example
(please read the docstring there). Requires mpi4py and, obviously, an
MPI installation.


####Contents of this dir####

signalflow_serverclient.png shows the flow of signals and data between
root (Server) to worker ranks (Clients). Only one rank i is shown, all
other ranks communicate in an identical fashion, and simultaneously,
with root. Solid arrows show the advancement of the program execution,
dashed arrows indicate communication between root and ranks (the
sending of signals and of data).

minimal.py is a trivial example of how to use the ServerClient class. Run:

 python minimal.py --help

for more info. 