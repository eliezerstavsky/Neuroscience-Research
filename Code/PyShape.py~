# Python Module for the Spectral Analysis of Shapes:

# Features:
#	Object-Oriented Design
#	Unit-Testing
#	ConDev

# Notes:
# Program does not currently support tetrahedra. Learn more about how to handle them.
# Program does not currently compute the volume of a closed surface. Learn about that.
# Reuter's key.txt must be in directory from which ipython was called.

# Imports:
import numpy as np
import vtk_operations as vo
import pyvtk
import subprocess
from time import time

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
	
	def __init__(self):
		'''Initialize attributes of shape object.'''
		self.Nodes = self.Mesh = self.Labels = 0
		self.has_nodes = self.has_mesh = self.has_labels = self.has_vtk = 0
		self.num_nodes = self.num_faces = 0
		
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
		
			self.has_nodes = self.has_mesh = 1
			self.num_nodes = self.Nodes.shape[0]
			self.num_faces = self.Mesh.shape[0]
			
			if Data.point_data.data != []:
				self.Labels = np.asarray(Data.point_data.data[0].scalars)
				self.has_labels = 1

			self.has_vtk = 1
		return 0
		
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
	
	############################################			
	# ------------------------------------------
	#     'Pre-Processing of Data' Methods      
	# ------------------------------------------
	
	def compute_mesh_measure(self):
		'''Computes the surface area of a shape object.'''
		
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
			
		verts = np.arange(self.num_nodes)
		meshing = set(self.Mesh.ravel())
		
		isolated = list(set.difference(verts, meshing))
	
		self.Nodes = np.delete(self.Nodes,isolated,0)
		self.num_nodes = self.Nodes.shape[0]
	
		# Update mesh numbering
		isolated = sorted(isolated, reverse=True)
		print isolated
		for i in isolated:
			for j in xrange(i, max_node+1):
				self.Mesh[self.Mesh==j] = j - 1
				
		self.num_faces = self.Mesh.shape[0]
		return 0
			
	def create_vtk(self, fname, label = 'Labels', header='Shape Analysis by PyShape'):
		'''Create vtk file from imported data.'''
		confirmed = 'y'
		if self.has_vtk:
			confirmed = raw_input('There is a vtk file containing the original data. Would you still like to create a new one?[y/n] ')
		
		if confirmed == 'y':
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

		return 
		
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
		
	############################################			
	# ------------------------------------------
	#     'Processing of Data' Methods      
	# ------------------------------------------
	
	#def compute_lbo(self,num=200): 
	#	'''Computation of the LBO using ShapeDNA_Tria software.'''
	#	p = subprocess.Popen('
	#if self.vtk == 0:
	#	self.vtk = create_vtk(self, fname

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
