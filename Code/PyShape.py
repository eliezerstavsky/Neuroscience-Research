# Python Module for the Spectral Analysis of Shapes:

'''
Features:
    Object-Oriented Design
    Unit-Testing
    ConDev

Notes:
    Program does not currently support tetrahedra. Learn more about how to handle them.
    Program does not currently compute the volume of a closed surface. Learn about that.
    Reuter's key.txt must be in directory from which ipython was called.
'''

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
	
	def __init__(self, id_tag='Testing'):
		'''Initialize attributes of shape object.'''
		self.id = str(id_tag)
		
		self.Nodes = self.Mesh = self.Labels = self.vtk = 0
		self.has_nodes = self.has_mesh = self.has_labels = self.has_vtk = 0
		self.num_nodes = self.num_faces = 0
		
		# For computing eigenspectrum of shape
		self.eigenvalues = self.eigenvectors = 0
		
		# For label propagation
		self.assigned_labels = 0
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
		# Check to make sure that data is inputted as a 2D array.
		# Check to make sure that meshing is polylines, triangles or tetrahedra
		
		mesh = np.asarray(mesh)
		if mesh.ndim !=2:
			print 'Please Enter Data as a 2D Array.'
		elif mesh.shape[1] < 2 or mesh.shape[1] > 4:
			print 'Please Provide Polylines, Triangles or Tetrahedra.'
		else:
			self.Mesh = mesh
			self.has_mesh = 1
			self.num_faces = self.Mesh.shape[0]
		return 0
		
	def add_labels(self, labels):
		'''Add labels to nodes as 1d array.'''
		# Check to make sure that labels are inputted as a 1D array. 
		
		labels = np.asarray(labels)
		if labels.ndim != 1:
			print 'Please enter labels as a 1D array.'
		else:
			self.Labels = np.asarray(labels)
			self.has_labels = 1
			self.assigned_labels = np.array(self.Labels.size)
		return 0
		
	def import_vtk(self, fname):
		'''Import all data from vtk file.'''
		# Check to make sure that fname is a string
		# Check to see if there are labels to be imported too
		
		if not isinstance(fname, str):
			print 'Please enter the file name as a string.'
		else:
			Data = pyvtk.VtkData(fname)
			self.Nodes = np.asarray(Data.structure.points)
			self.Mesh = np.asarray(Data.structure.polygons)
			
			if 'lh' in fname:
				self.id = 'lh' + Data.header
			elif 'rh' in fname:
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
		if new_nodes != self.Nodes:
			print 'There is a mismatch between the nodes in the fundus file and the nodes in the original file!'
		try:
			self.Fundi = np.asarray(Data.structure.lines)
			self.fundal_nodes = np.asarray(list(set(self.Fundi)))
		except:
			print 'The file does not contain polylines. Please import a different file.'
		
		if np.amax(self.Fundi) >= self.num_nodes:
			print 'The fundi reference nodes which are not in the file. Please consider.'
		
		return self.Fundi	

	def set_id(self, id_tag):
		'''Change the id_tag of the shape. Will be used to name files.'''
		self.id = str(id_tag)
		
	############################################			
	# ------------------------------------------
	#     'Access Data' Methods      
	# ------------------------------------------
	
	def get_num_nodes(self):
		''''Return the number of nodes in the shape object.'''
		return self.Nodes.shape[0]
	
	def get_num_faces(self):
		''''Return the number of faces in the shape object.'''
		return self.Mesh.shape[0]
	
	def get_nodes(self):
		'''Return the node list of the shape.'''
		return self.Nodes
		
	def get_faces(self):
		'''Return the face list of the shape.'''
		return self.Mesh
		
	def get_labels(self):
		'''Return the label list of the shape.'''
		return self.Labels		
	
	def get_assigned_labels(self):
		return self.assigned_labels
	
	def get_fundi(self):
		return self.Fundi
	
	def get_fundal_nodes(self):
		return self.fundal_nodes
	
	def get_id(self):
		'''Return the id_tag of the shape.'''
		return self.id		
	
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
				 'random' for preserving a <fraction> of random nodes.
		'''

		print 'Initializing labels by preserving {0} nodes.'.format(keep)
		
		preserved_labels = np.zeros(self.num_nodes)
			
		# If node is part of a fundus:
		if keep = 'fundi':
			preserved_labels = 
			
		# If node is part of triangle with a fundal nodes		
		if keep = 'border':
			for triangles in Meshes:
				node0, node1, node2 = triangles[0], triangles[1], triangles[2]
				num_nodes_in_fundi = (node0 in Fundi) + (node1 in Fundi) + (node2 in Fundi)
				if num_nodes_in_fundi > 0:
					preserved_labels[triangles] = 1
	
		preserved_labels[fundal_nodes == 1] = int(keep_fundi)
	
		# Change Labels:
		Labels[preserved_labels == 0] = -1
		num_changed = len(np.nonzero(preserved_labels == 0)[0])
	
		# OPTION 2. Just delete random nodes (keep 10%)
		if keep_random and not keep_fundi and not keep_attached:
			num_changed = 0
			for i in xrange(num_points):
				if np.mod(i,10) != 0:
					num_changed += 1
					Labels[i] = -1
			preserved_labels = np.ones(num_points)
			preserved_labels[Labels == -1] = 0
		
		results['num_changed'] = num_changed
		results['preserved_labels'] = preserved_labels
		results['Fundi_Labels'] = Labels
		
		print "num_changed:", num_changed
		print "percent_changed:", (num_changed+0.0)/num_points*100

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
	
# Derived Classes:

class ShapeRegions(Shape):
	'''
	
	'''
	
	def __init__(self):
		'''Establish important parameters for Analysis of Regions of Shapes.'''
		super(ShapeRegions, self).__init__()
		self.num_regions = []
	
