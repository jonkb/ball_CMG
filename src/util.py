""" Utility functions
"""

import time
import sympy as sp
from sympy.algebras.quaternion import Quaternion

"""
verbosity (int): How much to print
  0: Nothing
  1: Several status messages
  2: Pages of text
"""
verbosity = 1

def tic():
  """Start timing. Returns a list of times with one entry.
  I wrote these tic & toc functions for NMLAB
  See https://github.com/jonkb/NMLab/blob/main/src/util.py
  """
  times = []
  times.append(time.time())
  return times

def toc(times, msg=None, total=False):
  """Log the current time in the times list.
  If msg is provided, print out a message with the time.
    the string f" time: {t}" will be appended to the message.
  If total, then print out the elapsed time since the start.
    Else, print out only the last time interval.
  """
  times.append(time.time())
  if msg is not None:
    t = times[-1] - times[0] if total else times[-1] - times[-2]
    print(f"{msg} time: {t:.6f} s")

def sharp(v):
  """ 
  "Sharp" operator in Putkaradze paper
  Takes an R3 vector and turns it into a quaternion
  """
  return Quaternion(0, v[0], v[1], v[2])

def flat(q):
  """ 
  "Flat" operator in Putkaradze paper
  Takes a quaternion and turns it into an R3 vector
  """
  return sp.Matrix([[q.b], [q.c], [q.d]])

def printv(vlvl, *msgs):
  """ Simple print wrapper
  Prints if verbosity >= vlvl
  """
  if verbosity >= vlvl:
    print(*msgs)
