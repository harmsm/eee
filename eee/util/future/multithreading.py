"""
Classes and functions used to allow multithreading.
"""

import numpy as np
from tqdm.auto import tqdm

import multiprocessing as mp
import os


class MockLock():
    """
    Fake multiprocessing.Lock instance. Used when a function expects a lock but,
    for resource optimization purposes, we drop to one thread.
    """
    def __init__(self):
        pass

    def acquire(self):
        pass

    def release(self):
        pass


def _thread(args):
    """
    Run a function on a thread. Should only be called by thread_manager. Puts
    function return value and calculation number into the queue as a tuple.

    Parameters
    ----------
    args : tuple
        tuple with with calc number, function, kwargs, and queue

    Returns
    -------
    None
    """

    calc_number = args[0]
    fcn = args[1]
    kwargs = args[2]
    queue = args[3]

    out = fcn(**kwargs)

    # update queue
    queue.put((calc_number,out))


def thread_manager(kwargs_list,
                   fcn,
                   num_threads=None,
                   progress_bar=True,
                   pass_lock=False,
                   shared_kwarg=None):
    """
    Run a function multiple times in a mulithreaded fashion.

    Parameters
    ----------
    kwargs_list : list
        list of dictionaries holding kwargs to pass to function
    fcn : function
        python function to call with **kwargs
    num_threads : int
        number of threads to use for the calculation
    progress_bar : bool, default=True
        whether or not to use a tqdm progress bar
    pass_lock : bool, default=False
        pass a multiprocessing.Manager().Lock() instance to the function as a
        kwarg. (Function must know what to do with lock...)
    shared_kwarg : str, optional
        pass the kwarg indicated as a shared value to the function as a 
        multiprocessing.Value() or .Array() object. kwarg must point to a 
        float, int, float array, or int array. The function must deal with 
        locking. 

    Returns
    -------
    out : list
        list of outputs from the function calls, sorted by order in kwargs
    """

    # If num_threads not specified, figure out
    if num_threads is None:
        try:
            num_threads = mp.cpu_count()
        except NotImplementedError:
            num_threads = os.cpu_count()
        if num_threads is None:
            num_threads = 1

    
    # If only using one thread, don't waste overhead by making pool
    if num_threads == 1:

        if pass_lock:
            lock = MockLock()

        results = []
        if progress_bar:
            for kwargs in tqdm(kwargs_list):
                if pass_lock:
                    kwargs["lock"] = lock
                results.append(fcn(**kwargs))
        else:
            for kwargs in kwargs_list:
                if pass_lock:
                    kwargs["lock"] = lock
                results.append(fcn(**kwargs))

        return results


    manager = mp.Manager()
    queue = manager.Queue()
    lock =  manager.Lock()

    # If passing a shared kwarg
    if shared_kwarg is not None:

        try:
            to_share = kwargs_list[0][shared_kwarg]
        except KeyError:
            err = f"{shared_kwarg} not found in kwargs_list\n"
            raise ValueError(err)
        
        # Iterable, check for int or float and convert to Array
        if hasattr(to_share,"__iter__"):
            if np.issubdtype(int,to_share[0]):
                to_share = manager.Array("i",np.array(to_share,dtype=int))
            elif np.issubdtype(float,to_share[0]):
                to_share = manager.Array("d",np.array(to_share,dtype=float))
            else:
                err = "iterable must be float or int"
                raise ValueError(err)

        # Not iterable. Check for int or float and convert to Value
        else:
            if np.issubdtype(int,to_share):
                to_share = manager.Value("i",int(to_share))
            elif np.issubdtype(float,to_share):
                to_share = manager.Value("d",float(to_share))
            else:
                err = "shared value must be float or int\n"
                raise ValueError(err)

    all_args = []
    for i in range(len(kwargs_list)):

        # Append lock to kwargs if requested
        if pass_lock:
            kwargs_list[i]["lock"] = lock

        # Updated shared_kwarg to point to shared object if requested
        if shared_kwarg is not None:
            kwargs_list[i][shared_kwarg] = to_share

        all_args.append((i,fcn,kwargs_list[i],queue))

    with mp.Pool(num_threads) as pool:
        print("Running calculations.")

        # Black magic. pool.imap() runs a function on elements in iterable,
        # filling threads as each job finishes. (Calls _blast_thread
        # on every args tuple in all_args). tqdm gives us a status bar.
        # By wrapping pool.imap iterator in tqdm, we get a status bar that
        # updates as each thread finishes.
        if progress_bar:
            list(tqdm(pool.imap(_thread,all_args),total=len(all_args)))
        else:
            list(pool.imap(_thread,all_args))

    # Get results out of the queue.
    print("Assembling results.")
    results = []
    if progress_bar:
        with tqdm(total=queue.qsize()) as pbar:
            while not queue.empty():
                results.append(queue.get())
                pbar.update(1)
    else:
        while not queue.empty():
            results.append(queue.get())

    # Sort results
    results.sort()

    # Final results; a list holding the output of each function call
    results = [r[1] for r in results]

    return results
