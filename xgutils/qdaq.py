# modified from https://github.com/steven-lang/torch-utils
import torch.multiprocessing as mp
#import logging
from abc import ABC, abstractmethod
from typing import List, Union, Callable
from xgutils import sysutil
import os
import sys
import torch
import datetime
import traceback

# Get logger
#logger = logging.getLogger(__name__)


class Target():
    """
    Create target function that pulls an element from the queue and runs the
    job on that element.
    Args:
        job (Job): Job to run.
    Returns:
        Callable: Target to use for the multiprocessing API.
    """

    def __init__(self, job, logfile='multp.out'):
        self.job = job
        self.logfile = logfile

    def __call__(self, queue, conn):
        # Lock cuda device
        #cuda_device_id = queue.get()

        # Run job on this device
        cuda_device_id = conn.recv()
        try:
            self.job.run(cuda_device_id)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(
                f"Exception occured in CUDA {cuda_device_id}", file=sys.__stdout__)
            print(e, file=sys.__stdout__)
            with open(self.logfile, 'a') as f:
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=None, file=f)
                print(f"Exception occured in CUDA {cuda_device_id}", file=f)
                print(e, file=f)
        finally:
            # Free cuda device
            queue.put(cuda_device_id)
            print('Process %s finished' % (os.getpid()), file=sys.__stdout__)
            with open(self.logfile, 'a') as f:
                print('Process %s finished' % (os.getpid()), file=f)
        return True


class Job(ABC):
    """
    Abstract Job that describes what is to be done.
    The memberfunction `run` has to be implemented.
    """

    @abstractmethod
    def run(self, cuda_device_id):
        """
        Run the job on the given cuda device.
        Args:
            cuda_device_id (Union[int, List[int]]): Can be either a cuda device id or a list of 
                                                    cuda device ids.
        """
        pass


def start(jobs: List[Job], cuda_device_ids: Union[List[int], List[List[int]]], order=True):
    """
    Start the given jobs on the list of cuda devices.
    Args:
        jobs (List[Job]): List of jobs to start.
        cuda_device_ids (Union[List[int], List[List[int]]]):
            List of available cuda devices on which the jobs will be distributed.
            The list can either consist of single devices ([0, 1, 2, 3]) or device pairs
            ([[0, 1], [2, 3]]).
    """
    assert len(jobs) > 0, "No jobs specified."
    assert len(cuda_device_ids) > 0, "No cuda devices specified."
    stdout = sys.__stdout__
    # mp.set_start_method('spawn') # THIS SHOULD BE IN THE MAIN FILE
    # Create queue
    logfile = 'multip.out'
    with open(logfile, "a") as f:
        print('\n', file=f)
        print('############## NEW SESSION ###############', file=f)
        print('DATE: ', datetime.datetime.now(), file=f)
        print('\n', file=f)
    queue = mp.Queue(maxsize=len(cuda_device_ids))

    # Put all devices into the queue
    for id in cuda_device_ids:
        queue.put(id)
    # Create processes
    processes, connections = [], []
    for i, job in enumerate(jobs):
        f = Target(job, logfile=logfile)
        parent_conn, child_conn = mp.Pipe()
        connections.append(parent_conn)
        p = mp.Process(target=f, args=(queue, child_conn,))
        processes.append(p)
    # Start processes
    for i, p in enumerate(processes):
        p.start()
        with open(logfile, "a") as f:
            print(f"{i} Process {p.name} created, PID:{p.pid}", file=f)
    with open(logfile, "a") as f:
        print('- - - - -   Start Running - - - - -', file=f)
    for i, connection in enumerate(sysutil.progbar(connections)):
        cuda_device_id = queue.get()
        with open(logfile, "a") as f:
            print(
                f"{i} Process {processes[i].name} started, PID:{processes[i].pid}, CUDA:{cuda_device_id}", file=f)
        sys.stdout.flush()
        connection.send(cuda_device_id)

    # Join processes
    for p in processes:  # sysutil.progbar(processes):
        p.join()
    with open(logfile, "a") as f:
        print('############## SESSION ENDED ###############', file=f)


def test(name):
    print('hello', name)


class Foo(Job):
    def __init__(self, q):
        # Save some job parameters
        self.q = q

    def run(self, cuda_device_id):
        # Get cuda device
        device = torch.device(f"cuda:{cuda_device_id}")

        # Send data to device
        x = torch.arange(1, 4).to(device)
        # Compute stuff
        y = x.pow(self.q)
        print(f"Cuda device {cuda_device_id}, Exponent {self.q}, Result {y}")

        print('Sleeping ', 'PID:', os.getpid())
        import time
        import numpy as np
        time.sleep(np.random.rand()+1)


if __name__ == '__main__':
    exps = [Foo(i) for i in range(9)]
    start(exps, [1, 2, 3])
