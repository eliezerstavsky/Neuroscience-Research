# Python Module for the Spectral Analysis of Shapes:
####################################################
#
# Overview of Functions:
########################
#
# (A) Initialize Object **
#
# (B) Import Data
#  1- add_nodes **
#  2- add_mesh **
#  3- add_labels **
#  4- import_vtk **
#  5- import_fundi **
#  6- set_id **
#
# (C) Pre-Processing of Data
#  1- compute_mesh_measure **
#  2- compute_angles **
#  3- compute_smallest_angles **
#  4- remove_isolated **
#  5- refine_mesh **
#  6- fix_triangles **
#  7- initialize_labels (Done, check!)
#  8- check_well_formed ** 
#  9- create_vtk **
#  10- pre_process **
#
# (D) Processing of Data
#  1- compute_lbo
#  2- propagate_labels
#
# (E) Post-Processing of Data
#
# (F) Analysis of Data
#
# (G) Visualization of Data

###########################
# Imports:
import numpy as np
import networkx as nx
from scipy import sparse
import pyvtk
import subprocess
from time import time
import os
from subprocess import Popen, PIPE, check_output, STDOUT

import vtk_operations as vo
import compute_weights as cw
import graph_operations as go

np.set_printoptions(threshold='nan')

###################################
# Base Class:

class Shape:
	'''
	Shape Class. 
	1) Import data into object from either a vtk file or manually.
	2) Construct vtk if necessary.
	3) Pre-process data if necessary.
	4) Run LBO code.
	5) 
	'''
	
	# 'Initialize Object' Method
	
	def __init__(self, id_tag='PyShape Object'):
		'''Initialize attributes of shape object.'''
		self.id = str(id_tag)
		
		self.Nodes = self.Mesh = self.Labels = self.vtk = 0
		self.has_nodes = self.has_mesh = self.has_labels = self.has_vtk = 0
		self.num_nodes = self.num_faces = 0
		
		# For computing eigenspectrum of shape
		self.eigenvalues = self.eigenvectors = 0
		
		# For label propagation
		self.assigned_labels = 0
		self.preserved_labels = 0
		self.Fundi = self.fundal_nodes = 0
		self.border = 0
		
	############################################			
	# ------------------------------------------
	#     'Import Data' Methods      
	# ------------------------------------------
	
	def add_nodes(self, nodes):
		'''Add 3D coordinates of nodes as 2d array.'''
		# Check to make sure that Nodes were inputted as a 2D array.
		# Check to make sure that Nodes are of dimension <= 3
		
		nodes = np.asarray(nodes)
		if nodes.ndim != 2:
			print 'Please Enter Data as a 2D Array.'
		elif nodes.shape[1] > 3:
			print 'Please Provide Data of Dimension <= 3.'
		else:
			self.Nodes = nodes
			self.has_nodes = 1
			self.num_nodes = self.Nodes.shape[0]
		return 0	
		
	def add_mesh(self, mesh):
		'''Add triangular meshing as 2d array'''
		
		mesh = np.asarray(mesh)
		if mesh.ndim !=2:
			print 'Please Enter Data as a 2D Array.'
		elif mesh.shape[1] < 2 or mesh.shape[1] > 4:
			print 'Please Provide Polylines, Triangles or Tetrahedra.'
		elif not all([len(set(list(i)))==mesh.shape[1] for i in mesh]):
			print 'Some of your faces reference the same node multiple times.'
		elif np.amax(mesh) >=  self.num_nodes:
			print 'Meshing refers to non-existent nodes. Be advised.'
		else:
			self.Mesh = mesh
			self.has_mesh = 1
			self.num_faces = self.Mesh.shape[0]
		return 0
		
	def add_labels(self, labels):
		'''Add labels to nodes as 1d array.'''
		
		labels = np.asarray(labels)
		if labels.ndim != 1:
			print 'Please enter labels as a 1D array.'
		elif labels.shape[0] != self.num_nodes:
			print 'You have not provided the appropriate number of labels.'
		else:
			self.Labels = np.asarray(labels)
			self.has_labels = 1
			self.assigned_labels = np.array(self.Labels.size)
		return 0
		
	def import_vtk(self, fname, check=1):
		'''Import all data from vtk file.'''
		
		if not isinstance(fname, str):
			print 'Please enter the file name as a string.'
		else:
			Data = pyvtk.VtkData(fname)
			self.Nodes = np.asarray(Data.structure.points)
			self.Mesh = np.asarray(Data.structure.polygons)
			
			if 'lh' in fname:
				if check:
					conf = raw_input("Shall I change the id_tag to its region and number?[y/n]")
					if conf=='y':
						self.id = 'lh' + Data.header
			elif 'rh' in fname:
				if check:
					conf = raw_input("Shall I change the id_tag to its region and number?[y/n]")
					if conf=='y':
						self.id = 'rh' + Data.header
				
			self.has_nodes = self.has_mesh = 1
			self.num_nodes = self.Nodes.shape[0]
			self.num_faces = self.Mesh.shape[0]
			
			if Data.point_data.data != []:
				self.Labels = np.asarray(Data.point_data.data[0].scalars)
				self.has_labels = 1
				self.assigned_labels = np.array(self.Labels.size)

			self.has_vtk = 1
			self.vtk = open(fname, 'r')
			
		return self.id
	
	def import_fundi(self, fname):
		'''Import fundus lines from a vtk file.'''
		Data = pyvtk.VtkData(fname)
		
		new_nodes = np.asarray(Data.structure.points)
		if new_nodes.shape != self.Nodes.shape:
			print 'Nodes in the fundus file do not match nodes in the original file!'
		try:
			self.Fundi = np.asarray(Data.structure.lines)
			self.fundal_nodes = np.asarray(list(set(self.Fundi)))
		except:
			print 'The file does not contain polylines. Please import a different file.'
		
		if np.amax(self.Fundi) >= self.num_nodes:
			print 'The fundi reference nodes which are not in the file. Please consider.'
		
		return self.id	

	def set_id(self, id_tag):
		'''Change the id_tag of the shape. Will be used to name files.'''
		self.id = str(id_tag)
	
	############################################			
	# ------------------------------------------
	#     'Pre-Processing of Data' Methods      
	# ------------------------------------------
	
	def compute_mesh_measure(self, total=False):
		'''Computes the surface area of a shape object. 
		Finds the area of each triangle.'''
		
		# Check that nodes and meshing have been inputted.
		# Check that if shape is composed of polylines, the area is 0.
		# If shape is composed of tetrahedra, respond that method will not currently.
		
		if not(self.has_nodes and self.has_mesh):
			print 'Please input both the nodes and meshing of the shape.'
			return
		
		measure = np.zeros(self.Mesh.shape[0])
		if self.Mesh.shape[1] == 2:
			print 'The meshing comprises polylines. Length will be outputted.'
			i = 0
			for line in self.Mesh:
				measure[i] = np.linalg.norm(self.Nodes[line[0]] - self.Nodes[line[1]])
				i += 1
		elif self.Mesh.shape[1] == 3:
			print 'The meshing comprises triangles. Area will be outputted.'
			i = 0
			for triangle in self.Mesh:
				a = np.linalg.norm(self.Nodes[triangle[0]] - self.Nodes[triangle[1]])
				b = np.linalg.norm(self.Nodes[triangle[1]] - self.Nodes[triangle[2]])
				c = np.linalg.norm(self.Nodes[triangle[2]] - self.Nodes[triangle[0]])
				s = (a+b+c)/2.0
				
				measure[i] = np.sqrt(s*(s-a)*(s-b)*(s-c))
				i += 1
		elif self.Mesh.shape[1] == 4:
			print 'The meshing comprises tetrahedra. Computation currently unavailable.'
			measure = 0
		
		if total:
			return sum(measure)
		else:	
			return measure
	
	def compute_angles(self):
		'''Computes the angles for each triangle.'''
		# Currently only available for triangles.
		
		if self.Mesh.shape[1] != 3:
			print 'Sorry, this method only works for triangles.'
			return
			
		if not(self.has_nodes and self.has_mesh):
			print 'You have yet to add nodes and meshing!'
			
		angles = np.zeros((self.num_faces, 3))
		i = 0
		for triangle in self.Mesh:
			a = np.linalg.norm(self.Nodes[triangle[0]] - self.Nodes[triangle[1]])
			b = np.linalg.norm(self.Nodes[triangle[1]] - self.Nodes[triangle[2]])
			c = np.linalg.norm(self.Nodes[triangle[2]] - self.Nodes[triangle[0]])
			angles[i,0] = np.arccos((b**2+c**2-a**2)/(2.0*b*c))
			angles[i,1] = np.arccos((a**2+c**2-b**2)/(2.0*a*c))
			angles[i,2] = np.arccos((a**2+b**2-c**2)/(2.0*a*b))
			i += 1
		return angles
		
	def compute_smallest_angles(self, threshold=0.03):
		'''Find triangles in meshing with smallest angles.'''
		# Currently only available for triangles.
			
		if self.Mesh.shape[1] != 3:
			print 'Sorry, this method only works for triangles.'
			return
			
		if not(self.has_nodes and self.has_mesh):
			print 'You have yet to add nodes and meshing!'
			return
			
		angles = self.compute_angles()
		minima = np.amin(angles, 1)
		
		return np.arange(angles.size)[minima<threshold]
		
	def remove_isolated(self):
		'''Remove any vertices which are not connected to others via the meshing.'''
		# Remove any vertices which are not connected via meshing.
	
		if not(self.has_nodes and self.has_mesh):
			print 'You have yet to enter the nodes and meshing!'
			return
			
		verts = set(np.arange(self.num_nodes))
		meshing = set(self.Mesh.ravel())
		
		isolated = list(set.difference(verts, meshing))
	
		self.Nodes = np.delete(self.Nodes,isolated,0)
		self.num_nodes = self.Nodes.shape[0]
	
		# Update mesh numbering
		isolated = sorted(isolated, reverse=True)
		print isolated
		for i in isolated:
			for j in xrange(i, self.num_nodes+1):
				self.Mesh[self.Mesh==j] = j - 1
				
		self.num_faces = self.Mesh.shape[0]
		return 0
		
	def refine_mesh(self, depth = 1, which_fraction=1):
		'''Refine the meshing of the shape object.
		Option to refine only the largest triangles exists.
		Select which_fraction=float.
		Option to refine multiple times, by depth=int.
		'''
		# Check to make sure that the algorithm works for not only triangles!
		# Or if it does, document that.
		# Check that any time you change the number of nodes, you update the num_nodes attr.
		if not(self.has_nodes and self.has_mesh):
			print 'You have yet to enter the nodes and meshing!'
			return
		
		while(depth > 0):
		
			if which_fraction != 1:
				Areas = self.compute_mesh_measure()
				sortedIndex = np.argsort(Areas)
			else:
				sortedIndex = np.arange(self.num_faces)
		
			num_to_refine = which_fraction * self.num_faces
			old_triangles = []
			threshold = 1E-8
	
			for i in xrange(num_to_refine-1,-1,-1):
				# Find i'th largest triangle:
				ind = int(np.nonzero(sortedIndex==i)[0])

				# Record index of old triangle to delete later:
				old_triangles.append(ind)
	
				# Get vertices of this triangular mesh:		
				v0, v1, v2 = self.Nodes[self.Mesh[ind,0]], self.Nodes[self.Mesh[ind,1]], self.Nodes[self.Mesh[ind,2]]

				# Find midpoints of each edge:
				mid01 = (v0+v1)/2.0
				mid02 = (v0+v2)/2.0
				mid12 = (v1+v2)/2.0

				# Add vertices to the list of nodes:
				#############################################################
				# Check to make sure vertices aren't already in list of nodes:
				dist = self.Nodes - mid01 
				duplicates = [np.linalg.norm(dist[j]) for j in xrange(dist.shape[0])]
				minimum, minindex = np.amin(duplicates), np.argmin(duplicates)

				if minimum < threshold:
					# Duplicate! Assign new vertex the number of old vertex
					ind01 = minindex
				else:
					self.Nodes = np.vstack((self.Nodes,mid01))
					ind01 = self.Nodes.shape[0] - 1

				dist = self.Nodes - mid02 
				duplicates = [np.linalg.norm(dist[j]) for j in xrange(dist.shape[0])]
				minimum, minindex = np.amin(duplicates), np.argmin(duplicates)

				if minimum < threshold:
					# Duplicate! Assign new vertex the number of old vertex
					ind02 = minindex
				else:
					self.Nodes = np.vstack((self.Nodes,mid02))
					ind02 = self.Nodes.shape[0] - 1

				dist = self.Nodes - mid12 
				duplicates = [np.linalg.norm(dist[j]) for j in xrange(dist.shape[0])]
				minimum, minindex = np.amin(duplicates), np.argmin(duplicates)

				if minimum < threshold:
					# Duplicate! Assign new vertex the number of old vertex
					ind12 = minindex
				else:
					self.Nodes = np.vstack((self.Nodes,mid12))
					ind12 = self.Nodes.shape[0] - 1
				#############################################################

				# Add 4 new triangles:
				self.Mesh = np.vstack((self.Mesh,np.array([[self.Mesh[ind,0],ind01,ind02]])))
				self.Mesh = np.vstack((self.Mesh,np.array([[self.Mesh[ind,1],ind01,ind12]])))
				self.Mesh = np.vstack((self.Mesh,np.array([[self.Mesh[ind,2],ind02,ind12]])))
				self.Mesh = np.vstack((self.Mesh,np.array([[ind12,ind01,ind02]])))

			# Delete triangles which were refined:
			for old in sorted(old_triangles, reverse=True):
				self.Mesh = np.delete(self.Mesh,[old*3,old*3+1,old*3+2]).reshape(-1,3)

			self.num_nodes = self.Nodes.shape[0]
			self.num_faces = self.Mesh.shape[0]
			
			depth -= 1

		return 0
		
	def fix_triangles(self, method='delete', threshold=0.03):
		'''Handle the ill-shaped triangles of low quality.
		First attempt: delete the triangles.'''
		
		if not(self.has_nodes and self.has_mesh):
			print 'You have yet to enter the nodes and meshing!'
			return
		
		# Find triangles with angles below the threshold.
		low_quality = self.compute_smallest_angles()
		
		if method=='delete':
			# Delete those triangles from the meshing.
			bad_triangles = sorted(low_quality, reverse=True)
			for t in bad_triangles:
				self.Mesh = np.delete(self.Mesh, t, 0)
			self.num_faces = self.Mesh.shape[0]
			
		return sorted(low_quality)
	
	def initialize_labels(self, keep='border', fraction=.05):
		'''Initialize a set of labels to serve as the seeds for label propagation.
		Options include: 'border' for nodes connected to fundi.
		                 'fundi' for nodes which are part of fundi.
				 		 'both' for both the fundi and the borders.
				 		 'random' for preserving a <fraction> of random nodes.'''
		
		if not self.has_labels:
			print 'The object does not have any labels. Please add them.'
			return
			
		print 'Initializing labels by preserving {0} nodes.'.format(keep)
		self.assigned_labels = np.zeros(self.num_nodes) - 1
		self.preserved_labels = np.zeros(self.num_nodes)
			
		# To preserve the fundi nodes, find all nodes which are part of a fundus, and
		# record their index in the array self.preserved_labels
		if keep in ['fundi','both']:
			preserved_labels[self.fundal_nodes] = 1
			
		# To preserve the border nodes, find all nodes which are part of a triangle with
		# fundal nodes, and record their index in the array self.preserved_labes 		
		if keep in ['border','both']:
			for triangles in self.Mesh:
				node0, node1, node2 = triangles[0], triangles[1], triangles[2]
				num_nodes_in_fundi = (node0 in Fundi) + (node1 in Fundi) + (node2 in Fundi)
				if num_nodes_in_fundi > 0:
					self.preserved_labels[triangles] += 1
			self.preserved_labels[self.fundal_nodes] -= 1
		
		# To preserve a fraction of random nodes, keep every 1/fraction'th label.
		if keep == 'random':
			if fraction > 1:
				print 'Please enter a fractional number less than or equal to 1.'
				return
			
			randoms = np.array([np.mod(i, int(1.0/fraction)) for i in xrange(num_nodes)])
			self.preserved_labels[randoms==0] = 1
		
		# Assign the preserved labels to self.assigned_labels.
		self.assigned_labels[self.preserved_labels==1] = self.Labels[self.preserved_labels==1]
		
		# self.assigned_labels now contains the 'true' labels of a subset of the nodes.
		# main work of the function is complete.
		# now, provide some statistics for what was done.
		
		self.num_labels_preserved = len(self.preserved_labels[self.preserved_labels==1])
		self.percent_labels_preserved = (self.num_labels_preserved+0.0)/self.num_nodes * 100
		
		print 'The percentage of preserved labels: {0}'.format(self.percent_labels_preserved)

		return self.assigned_labels
	
	def get_label_matrix(self, Labels):
		'''Constructs an n x C matrix of labels. 
		Input: Array of n labels. -1 corresponds to no label.
		Output n x C matrix. Row corresponds to node, column corresponds to class. 
		1 in column = membership in that class. -1 = absence. 0 = unlabeled data.'''
		
		# Remove duplicates
		set_of_labels = np.sort(np.asarray(list(set(Labels)))) 

		# If all data is labeled, insert -1 at beginning of list for consistency of later methods.
		if -1 not in set_of_labels:		
			set_of_labels = np.insert(set_of_labels, 0, -1)

		# Number of classes and nodes
		C = len(set_of_labels) - 1	
		n = Labels.shape[0]

		# Relabel the classes to 0 through C, 0 now indicating no class.
		for i in set_of_labels[2:]:
			Labels[np.nonzero(Labels == i)] = np.nonzero(set_of_labels == i)
		Labels[np.nonzero(Labels == 0)] = 1
		Labels[np.nonzero(Labels == -1)] = 0

		# Create a dictionary mapping new class labels to old class labels:
		self.label_mapping = dict([(i, set_of_labels[i+1]) for i in xrange(-1,C)])
		print "Label Mapping:", self.label_mapping
	
		# Construct L x C Matrix	
		self.label_matrix = np.zeros((n, C))
	
		for i in xrange(n):
			if Labels[i] != 0:
				self.label_matrix[i, :] = -1
				self.label_matrix[i, Labels[i] - 1] = 1
	
		return self.label_matrix, self.label_mapping
	
	def check_well_formed(self):
		'''Check whether the inputted data is well formed.'''
		# Check that number of labels corresponds to number of nodes.
		# Check that numbers in meshing don't exceed number of nodes.
		
		if not self.has_nodes:
			print 'There are no nodes!'
		if not self.has_mesh:
			print 'There are no faces!'
		
		if self.has_labels and self.has_nodes:
			if self.Labels.size != self.num_nodes:
				print 'There is a mismatch betweeen the number of labels provided \
						and the number of nodes in the shape.'
				print 'There are {0} nodes and {1} labels. Please fix'.format(self.Nodes.shape[0],self.Labels.size)
		
		if self.has_nodes and self.has_labels:
			max_mesh_num = np.amax(self.Mesh)
			if max_mesh_num >= self.num_nodes:
				print 'The meshing contains reference to a non-existent node. Please fix.'
		
		return 0

	def create_vtk(self, fname, label = 'Labels', header='Shape Analysis by PyShape'):
		'''Create vtk file from imported data.'''
		if not(self.has_nodes and self.has_mesh):
			print 'You have yet to enter the nodes and meshing!'
			return
			
		if not self.has_labels:
			self.Labels = None
			
		self.vtk = vo.write_all(fname, self.Nodes, self.Mesh, self.Labels, label_type=label, msg=header)
		print 'vtk file was successfully created at: ', self.vtk.name
		self.has_vtk = 1
		
		return self.vtk.name

	def pre_process(self, fname):
		'''Full pre-processing of the shape object.'''
		self.remove_isolated()
		self.fix_triangles()
		self.check_well_formed()
		self.create_vtk(fname)
	
	############################################			
	# ------------------------------------------
	#     'Processing of Data' Methods      
	# ------------------------------------------
	
	def compute_lbo(self, num=500, check=0, fname='/home/eli/Neuroscience-Research/Analysis_Hemispheres/Testing.vtk'): 
		'''Computation of the LBO using ShapeDNA_Tria software.'''
		# Check that everything has been done properly
		# Create vtk file
		
		if not(self.has_nodes and self.has_mesh):
			print 'You have yet to enter the nodes and meshing!'
			return
		
		proceed = 'y'
		if check:
			proceed = raw_input('Has the data been pre-processed?[y/n] ')
		
		if proceed == 'n':
			print 'Then consider running check_well_formed(), remove_isolated(), fix_triangles()'
			return
		
		if self.has_vtk == 0:
			print 'Creating a vtk file for visualization and data processing'
			self.vtk = create_vtk(self, fname)
		
		# Run Reuter's code:
		outfile = fname[:-4]+'_outfile'
		execute = str('./shapeDNA-tria/shapeDNA-tria --mesh ' + self.vtk.name + ' --num ' + str(num) + ' --outfile ' + outfile + ' --ignorelq')
		params = ' --mesh ' + self.vtk.name + ' --num ' + str(num) + ' --outfile /home/eli/Desktop/outfile_' + self.id
		
		process = Popen(execute, shell=True, stdout = PIPE, stderr = STDOUT)
		# print repr(process.communicate()[0])
		if self.num_nodes < 5000:
			time.sleep(7)
		else:
			time.sleep(16)	
		f = open(outfile)
		
		eigenvalues = np.zeros(num)
		add = False
		i = 0
				
		for line in f:
			if add:
				line = line.strip('{')
				line = line.replace('}','')
				line = line.replace('\n','')
				vals = line.split(';')
				if vals[-1] is '':
					vals.pop()
				try:
					vals = map(float,vals)
				except:
					vals = [-1]
					print 'Could not properly convert line'
					
				eigenvalues[i:i+len(vals)] = vals
				i += len(vals)
			elif 'Eigenvalues' in line:
				add = True
				
			if i == num:
				break
		
		self.eigenvalues = eigenvalues
		return self.eigenvalues
	
	def propagate_labels(self,method='weighted_average', kernel=cw.rbf_kernel, sigma=1, alpha=1, diagonal=0, repeat=1, max_iters=1000, tol=1e-3, eps=1e-7):
		'''Main function to propagate labels.
		A number of methods may be used for this purpose:
		1) 'weighted_average'
		2) 'propagation'
		3) 'spreading'
		The function takes the following parameters:
		1) method - choice of algorithm
		2) kernel - for use in constructing affinity matrix
		3) sigma - parameter for gaussian kernel
		4) alpha - clamping factor (0<a<1)
		5) repeat - number of separate times to run algorithm
		6) max_iters - number of times to iterate an algorithm
		7) tol - threshold to assess convergence
		8) eps - for numerical stability
		9) diagonal - option to change values along the diagonal, will have an effect on some alg.'''
		
		# Step 1. Construct Affinity Matrix - compute edge weights:
		self.aff_mat = cw.compute_weights(self.Nodes,self.Mesh,kernel=kernel,sigma=sigma)
		
		# Step 2. Transform column of labels into L x C Matrix, one column per class
		a,b = self.get_label_matrix(self.Labels)
		
		# Step 3. Propagate Labels!
		if method == "weighted_average":
			print 'Performing Weighted Average Algorithm! Parameters: max_iters={0}'.format(str(max_iters))
			(Graph, best_guess) = self.weighted_average(repeat, diagonal, max_iters, tol)
		
		elif algorithm == "Label_Spreading":
			print 'Performing Label Spreading Algorithm! Parameters: alpha={0}, max_iters={1}'.format(str(alpha), str(max_iters))
			Graph = label_spreading(G, Label_Matrix, aff_mat, label_mapping, data['num_changed'],
									repeat, alpha, max_iters, tol)
		elif algorithm == "Label_Propagation":
			Graph = label_propagation(G, Label_Matrix, aff_mat, label_mapping, data['num_changed'],
								 	  repeat, alpha, eps, max_iters, tol)
		elif algorithm == "Nearest_Neighbor":
			print 'Performing 1_nearest_neighbor algorithm!'
			Graph = nearest_neighbor(G, Label_Matrix, label_mapping)
		else:
			Graph = "That algorithm is not available."
			
	############################################			
	# ------------------------------------------
	#     'Post-Processing of Data' Methods      
	# ------------------------------------------
	
	############################################			
	# ------------------------------------------
	#     'Analysis of Data' Methods      
	# ------------------------------------------
	
	############################################			
	# ------------------------------------------
	#     'Visualization of Data' Methods      
	# ------------------------------------------
	
	############################################
	# ------------------------------------------
	# 			  Helper Methods
	# ------------------------------------------
	
	def weighted_average(self, repeat, diagonal_entries, max_iters, tol):
		'''Performs iterative weighted average algorithm to propagate labels to unlabeled nodes.
		Features: Hard label clamps, deterministic solution.
		See: Zhu and Ghahramani, 2002.'''
        
        '''The first approach to be considered in the semi-supervised learning case it to propagate labels on the graph. A simple algorithm of this sort has been propoosed by Zhu and Ghahramani (2002), and starts (like all such algorithms) with a set of n nodes, l of which are labeled, and u unlabeled. The algorithm takes as its input the affinity matrix W (self.aff_mat). From the affinity matrix, one may construct the diagonal degree matrix, which is a measure of the total weight (or number of edges) which are attached to a node.'''
        
        self.DDM = go.compute_diagonal_degree_matrix(self.aff_mat, inverse=True)
        
        '''Next, we must initialize a vector to represent the results of the label propagation algorithm. It will contain l labels and u 0's.This has already been done by the function initialize_labels, and is called self.assigned_labels. We will just check to make sure this has been accomplished.'''
        
        if self.assigned_labels == 0:
            print 'Please initialize the labels by calling self.initialize_labels()'
            return
        
        '''Now, we can actually proceed to perform the iterative algorithm. At each timestep, self.assigned_labels will be updated to reflect the weighted average of adjacent nodes. An important caveat of this algorithm, is that the labeled nodes remain fixed, or clamped. They should not be changed, and will need to be reset. We accomplish the reset by recalling that self.preserved_labels stores the indexes of those nodes whose labels were preserved, and self.Labels contains the actual labels.
        The algorithm repeates itself until either convergence or max_iters (which will prevent excessive computation time).
        We must also take care to solve the multi-label problem. To do so, we employ a one-vs-all framework, where each label is considered independently, and set against the rest of the labels.
        More specifically, '''
        
        converged = False
        while not converged and max_iters > 0:
            tmp = np.mat(self.DDM * self.aff_mat * np.mat(self.assigned_labels))
            converged = np.linalg.norm(self.assigned_labels - tmp) < tol
            self.assigned_labels[self.preserved_labels==1] = self.Labels[self.preserved_labels==1]
        
            max_iters -= 1
        
        restore = max_iters
	
		# Affinity matrix W has diagonal entries of 0. They can be changed in the following loop.
		if diagonal_entries != 0:
			for i in xrange(num_nodes):
				self.aff_mat[i, i] = diagonal_entries
	
		# Construct inverse of Diagonal Degree Matrix
		self.diag_degree_inv = go.compute_diagonal_degree_matrix(self.aff_mat, inverse=True)
	
		# Allow for the option to iteratively do the algorithm:
		# i.e. it will rerun if the number of unlabeled_nodes exceeds a threshold
		threshold_unlabeled = 100
		number_unlabeled = num_changed
	
		# In the one vs. all framework, let all terminally unlabeled nodes be labeled with best guess.
		# Actually, let each node's guesses be recorded.
		# Then, afterwards, check which nodes were labeled incorrectly.
		# For a node which was labeled incorrectly, find how close you were to getting it.
		# Let a node be labeled with multiple classes?	
	
		best_guess = np.zeros((n,Label_Matrix.shape[1]))		
	
		while (number_unlabeled > threshold_unlabeled) and (repeat > 0):
			# In one vs. all fashion, iteratively process the weighted average of neighbors.	
			column_index = 0
			for column in Label_Matrix.T:	
				mat_column = np.mat(column).T
				labeled_indices = np.nonzero(mat_column != 0)[0]
				num_members = (np.nonzero(mat_column == 1))[0].shape[1]
				print 'Initial number of members in class {0}: {1}'.format(str(column_index), str(num_members))  
			
				Y_hat_now = mat_column
				Y_hat_next = diag_degree_inv * aff_mat * Y_hat_now
		
				while not_converged(Y_hat_next, Y_hat_now, tol) and max_iters > 0:
					Y_hat_now = Y_hat_next
					Y_hat_next = diag_degree_inv * aff_mat * Y_hat_now
					Y_hat_next[labeled_indices] = mat_column[labeled_indices]
					max_iters -= 1
		
				# Store results of algorithm in best guess for use later.
				best_guess[:,column_index] = np.asarray(Y_hat_next).T
			
				class_members = np.nonzero(Y_hat_next>0)[0]
				print 'Current number of members in class {0}: {1}'.format(str(column_index), str(class_members.shape[1]))
			
				Label_Matrix[class_members, :] = -1
				Label_Matrix[class_members, column_index] = 1
				
				column_index += 1
				max_iters = restore
		
			# Account for the case where an unlabeled node does not get labeled to any class:
			number_unlabeled = 0
			node_num = 0
			for line in Label_Matrix:
				if 0 in line:
					number_unlabeled += 1
					Label_Matrix[node_num, :] = -1
					best = np.argmax(best_guess[node_num,:])
					Label_Matrix[node_num, best] = 1
				node_num += 1				 
			print "number of nodes that were unlabeled:", number_unlabeled
		
			repeat -= 1
				
		Graph_Final = label_nodes(G, Label_Matrix, label_mapping)

		return Graph_Final, best_guess	
	
# Derived Classes:

class ShapeRegions(Shape):
	'''
	
	'''
	
	def __init__(self):
		'''Establish important parameters for Analysis of Regions of Shapes.'''
		super(ShapeRegions, self).__init__()
		self.num_regions = []
	
