# Python Script for Analyzing Hemispheric Asymmetry!

# Features:

# Notes:

# Imports:
import numpy as np
import PyShape
from time import time
import os
import pylab

# Parameters:
num_eigs = 500

# ----------------------------------------
# Compile list of vtk files to be analyzed
# ----------------------------------------

path = '/home/eli/Neuroscience-Research/Inflated_Hemispheres_KKI2009_15/'
files = os.listdir(path)
files.remove('lh_8.vtk')
files.remove('rh_8.vtk')
num_files = len(files)
high = 37

# ---------------------------------------------------
# Create shape objects for each file and process them
# ---------------------------------------------------

Eigenvalues = np.zeros((high*2, num_eigs))
counter = 0
for f in files:
	print 'Processing file number %d' % counter
	counter += 1
	if '~' not in f and '.vtk' in f:
		shape = PyShape.Shape()
		id_tag = shape.import_vtk(path+f)
		shape.pre_process(path+'Analysis/'+id_tag+'.vtk')
		eigs = shape.compute_lbo(fname=str(path+'Analysis/'+id_tag+'.vtk')) * (shape.compute_mesh_measure(total=True)**(2.0/3.0))
		
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
for line1 in Eigenvalues[:high, :]:
	if sum(line1) != 0:
		for line2 in Eigenvalues[high:,:]:
			if sum(line2) != 0:
				distance[i,j] = np.linalg.norm(line1 - line2) 
			j += 1
	i +=1
	j = 0
	
# ------------------------------
# Display results of computation
# ------------------------------

mins = np.zeros(high)
i=0
for eachline in distance:
	print eachline
	eachline[eachline==0] = 1000
	minimum = np.argmin(eachline)
	mins[i] = minimum
	i+=1
	
pylab.plot(mins)
