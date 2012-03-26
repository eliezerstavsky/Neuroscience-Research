''' Graph-Based Label Propagation Algorithm for Semi-Supervised Learning.

Model Features:
--------------
Label Clamping: Hard, Soft
Multiple Kernels: Gaussian, kNN
Multiple Algorithms: "Weighted Average", "Label_Spreading", "Label_Propagation"
'''

# Imports:
import numpy as np
import networkx as nx
from scipy import sparse
from time import time
import pyvtk

import vtk_operations as vo
import compute_weights as cw
import graph_operations as go

np.set_printoptions(threshold='nan')
