# Python Script for Analyzing Hemispheric Asymmetry!

# Features:

# Notes:

# Imports:
import numpy as np
import PyShape
from time import time
import os

# Parameters:
num_eigs = 200

# ----------------------------------------
# Compile list of vtk files to be analyzed
# ----------------------------------------

path = '/home/eli/Neuroscience-Research/Data_Hemispheres/'
files = os.listdir(path)
files.remove('KKI2009_15_lh_30.vtk')
files.remove('KKI2009_15_rh_30.vtk')
num_files = len(files)
high = 37

# ---------------------------------------------------
# Create shape objects for each file and process them
# ---------------------------------------------------

Eigenvalues = np.zeros((high*2, num_eigs))

for f in files:
	t0 = time()
	if '~' not in f and '.vtk' in f:
		shape = PyShape.Shape()
		id_tag = shape.import_vtk(path+f)
		shape.pre_process()
		eigs = shape.compute_lbo()
		
		if 'lh' in id_tag:
			n = id_tag.strip('lh')
			Eigenvalues[int(n),:] = eigs
			
		elif 'rh' in id_tag:
			n = id_tag.strip('rh')
			Eigenvalues[int(n)+high,:] = eigs
	
# -------------------------------------------------
# Compute metric between regions of each hemisphere
# -------------------------------------------------

distance = np.zeros((high, high))

i,j = 0,0
for line1 in Eigenvalues[:high]:
	if sum(line1) != 0:
		for line2 in Eigenvalues[high:]:
			if sum(line2) != 0:
				distance[i,j] = np.linalg.norm(line1 - line2) 
			j += 1
	i += 1
	
# ------------------------------
# Display results of computation
# ------------------------------

print distance


