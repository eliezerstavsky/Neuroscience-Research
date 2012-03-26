""" Graph-Based Label Propagation Algorithms for Semi-Supervised Learning.

Model Features
--------------
Label Clamping: Hard, Soft
Multiple Kernels: Gaussian, kNN
Multiple Algorithms: "Weighted_Average" (default), "Label_Spreading", "Label_Propagation"
"""

import numpy as np
import networkx as nx
from scipy import sparse
import time
import pyvtk

from vtk_operations import *
from compute_weights import *
from graph_operations import *
# from kernels import rbf_kernel
# from sklearn.neighbors.graph import *

np.set_printoptions(threshold='nan')

# Default Values
default_sigma = 10
default_alpha = 1
default_kernel = rbf_kernel
default_repeat = 1
default_algorithm = "Weighted_Average"
default_VTK ='/home/eli/PythonFiles/Data/testdatalabels.vtk'
default_fundi_file ='/home/eli/PythonFiles/Data/testdatafundi.vtk' 

### Main Function:
def LabelPropagation(VTK = default_VTK, fundi_file = default_fundi_file, algorithm=default_algorithm, kernel=default_kernel, sigma=default_sigma, alpha=default_alpha, diagonal=0, 
					 repeat=default_repeat, max_iters=3000, tol=1e-3, eps=1e-7):
	"""
	Main Function: Propagates labels to unlabeled nodes in graph.
	
	Input: 	VTK file containing:
				n x 3 Matrix X of 3-dimensional nodes.
					n = l + u: l labeled nodes, u unlabeled nodes
				m x 3 Matrix M of triangular meshes.
				n x 1 Array of C unique labels.
					C = number of classes
					0 = no label
				
	Output:	Graph G with n labeled nodes along with probabilities.
 	
	Parameters
	----------
	kernel: function (x1, x2) --> float
		choice of similarity metric
	sigma: float
		parameter for gaussian (rbf) kernel
	
	alpha: float
		clamping factor
	
	diagonal: float
		choice of diagonal entries in weight matrix
	
	max_iters: float
		maximum number of iterations allowed
	tol: float
		threshold to consider the system at steady state
		
	eps: float
		epsilon value for numerical stability in some algorithms.
	"""
	
	Nodes = Meshes = Labels = A = 0
	
	# Step 0. Convert VTK file into numpy matrices
	Data = pyvtk.VtkData(VTK)
	Nodes = Data.structure.points
	Meshes = Data.structure.polygons
	
	data = convert_vtk_to_matrix(VTK, fundi_file)
	
	# Step 1. Transform input data into graph G_basic
	G_basic = build_graph(data['Nodes'])

	# Step 2. Compute edge weights using connectivity established in VTK file
	(G, aff_mat) = compute_weights(data['Nodes'], data['Meshes'], kernel, G=G_basic, sigma=sigma)
	aff_mat = aff_mat.tocsr()
	
	# Step 3. Transform column of labels into L x C Matrix, one column per class
	(Label_Matrix, label_mapping) = get_label_matrix(data['Fundi_Labels'])

	# Step 4. Propagate Labels! 
	if algorithm == "Weighted_Average":
		print 'Performing Weighted Average Algorithm! Parameters: max_iters={0}'.format(str(max_iters))
		(Graph, best_guess) = weighted_average(G, Label_Matrix, aff_mat, label_mapping, data['num_changed'],
								 repeat, diagonal, max_iters, tol)
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

	# Step 5. Assess success of algorithm:
	results = assess(Graph, data['Manual_Labels'], data['num_changed'], data['preserved_labels'], best_guess, label_mapping)
	
	h = open('/home/eli/Desktop/NewLabels.vtk', 'w')
	line_num = 0
	total_lines_to_copy = 9 + data['Nodes'].shape[0] + data['Meshes'].shape[0] 
	for line in data['main_file']:
		if line_num < total_lines_to_copy:
			h.write(line)
			line_num += 1
	for i in xrange(len(Graph.nodes())):
		h.write('{0}\n'.format(str(results[i])))
	
	h.close()
		
	return Graph, results
	
### Algorithm Functions:

def weighted_average(G, Label_Matrix, aff_mat, label_mapping, num_changed,
					 repeat, diagonal_entries, max_iters, tol):
	"""Performs iterative weighted average algorithm to propagate labels to unlabeled nodes.
	Features: Hard label clamps, deterministic solution.
	See: Zhu and Ghahramani, 2002.
	"""
	restore = max_iters
	n = aff_mat.shape[0]
	
	# Affinity matrix W has diagonal entries of 0. They can be changed in the following loop.
	if diagonal_entries != 0:
		for i in xrange(n):
			aff_mat[i, i] = diagonal_entries
	
	# Construct inverse of Diagonal Degree Matrix
	diag_degree_inv = compute_diagonal_degree_matrix(aff_mat, inverse=True)
	
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
		
def label_spreading(G, Label_Matrix, aff_mat, label_mapping, num_changed, 
					repeat, alpha, max_iters, tol):
	""" Performs label spreading algorithm on nodes.
	Features: Soft label clamps, deterministic solution.
	See: Zhou et al., 2004.
	"""
	print 'algorithm is starting'
	restore = max_iters
	n = aff_mat.shape[0]

	# Allow for the option to iteratively do the algorithm:
	# i.e. it will rerun if the number of unlabeled_nodes exceeds a threshold
	threshold_unlabeled = 100
	number_unlabeled = num_changed
	print 'about to compute the normalized laplacian'
	ddm = compute_diagonal_degree_matrix(aff_mat, inverse=False, square_root=False)
	Laplacian = ddm - aff_mat
	print 'Laplacian, unnormalized:', Laplacian
	ddi_sq = compute_diagonal_degree_matrix(aff_mat, inverse=True, square_root=True)
	print 'ddi_sq:', ddi_sq[0:10, 0:10].todense()
	L = ddi_sq * Laplacian * ddi_sq
	print 'finished computing the normalized laplacian:', L[0:10,0:10].todense()
	
	while (number_unlabeled > threshold_unlabeled) and (repeat > 0):
		# In one vs. all fashion, iteratively process the weighted average of neighbors.	
		column_index = 0
		for column in Label_Matrix.T:	
			mat_column = np.mat(column).T
			labeled_indices = np.nonzero(mat_column != 0)[0]
			num_members = (np.nonzero(mat_column == 1))[0].shape[1]
			print 'Initial number of members in class {0}: {1}'.format(str(column_index), str(num_members))  

			Y_hat_now = mat_column
			Y_hat_next = alpha * L * Y_hat_now + (1 - alpha) * mat_column

			while not_converged(Y_hat_next, Y_hat_now, tol) and max_iters > 0:
				Y_hat_now = Y_hat_next
				Y_hat_next = alpha * L * Y_hat_now + (1 - alpha) * mat_column
				max_iters -= 1

			class_members = np.nonzero(Y_hat_next>0)[0]
			print 'Current number of members in class {0}: {1}'.format(str(column_index), str(class_members.shape[1]))
			
			Label_Matrix[class_members, :] = -1
			Label_Matrix[class_members, column_index] = 1

			column_index += 1
			print 'max_iters', max_iters
			max_iters = restore

		repeat -= 1
		
		# Account for the case where an unlabeled node does not get labeled to any class:
		number_unlabeled = 0
		for line in Label_Matrix:
			if 0 in line:
				number_unlabeled += 1 
		print "number of nodes still unlabeled:", number_unlabeled

	Graph_Final = label_nodes(G, Label_Matrix, label_mapping)

	return Graph_Final


# THE following algorithm is not yet properly written.
def label_propagation(G, Label_Matrix, aff_mat, label_mapping, num_changed, 
					  repeat, alpha, eps, max_iters, tol):
	"""Performs iterative weighted average algorithm to propagate labels to unlabeled nodes.
	Features: Hard label clamps, deterministic solution.
	See: Zhu and Ghahramani, 2002.
	"""
	restore = max_iters
	n = aff_mat.shape[0]

	# Construct Diagonal Degree Matrix
	diag_degree = compute_diagonal_degree_matrix(aff_mat, inverse=True)
	
	# Variables necessary for the algorithm:
	mu = alpha / (1 - alpha + eps)
	A = sparse.lil_matrix((n, n))
	A = A.tocsr()
	A.setdiag(1 / (1 + mu*diag_degree + mu*eps))
	np.identity(n) + mu*diag_degree + mu*eps

	W = W.tocsr()
	if no_inverse == False:	
		ddmi.setdiag(1 / W.sum(axis=1))	
	else:
		ddmi.setdiag(W.sum(axis=1))		
	ddmi = ddmi.tocsr()

	# Allow for the option to iteratively do the algorithm:
	# i.e. it will rerun if the number of unlabeled_nodes exceeds a threshold
	threshold_unlabeled = 100
	number_unlabeled = num_changed

	while (number_unlabeled > threshold_unlabeled) and (repeat == 1):
	# In one vs. all fashion, iteratively process the weighted average of neighbors.	
		column_index = 0
		for column in Label_Matrix.T:	
			mat_column = np.mat(column).T
			labeled_indices = np.nonzero(mat_column != 0)[0]
			num_members = (np.nonzero(mat_column == 1))[0].shape[1]
			print 'Initial number of members in class {0}: {1}'.format(column_index, num_members)  

			Y_hat_now = mat_column
			Y_hat_next = np.inv(A) * (mu * W * Y_hat_now + column)

			while not_converged(Y_hat_next, Y_hat_now, tol) and max_iters > 0:
				Y_hat_now = Y_hat_next
				Y_hat_next = diag_degree_inv * aff_mat * Y_hat_now
				Y_hat_next[labeled_indices] = mat_column[labeled_indices]
				max_iters -= 1

			class_members = np.nonzero(Y_hat_next>0)[0]
			print 'Current number of members in class {0}: {1}'.format(column_index, class_members.shape[1])

			Label_Matrix[class_members, :] = -1
			Label_Matrix[class_members, column_index] = 1

			column_index += 1
			max_iters = restore

		# Account for the case where an unlabeled node does not get labeled to any class:
		number_unlabeled = 0
		for line in Label_Matrix:
			if 0 in line:
				number_unlabeled += 1 
		print "number of nodes still unlabeled:", number_unlabeled
		repeat = 0

	Graph_Final = label_nodes(G, Label_Matrix, label_mapping)

	return Graph_Final
	
### Helper Functions:

def build_graph(Nodes):
	"""Construct networkx graph of inputted data."""
	
	G = nx.Graph()
	index = 0
	
	for coordinates in Nodes:
		G.add_node(index, position = coordinates)
		index += 1
	return G
	
def get_label_matrix(Labels):
	"""Constructs an n x C matrix of labels. 
	
	Input: Array of n labels. -1 corresponds to no label.
	Output n x C matrix. Row corresponds to node, column corresponds to class. 
	1 in column represents membership in that class. -1 represents absence. 0 represents unlabeled data.
	"""
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
	label_mapping = dict([(i, set_of_labels[i+1]) for i in xrange(-1,C)])
	print "Label Mapping:", label_mapping
	
	# Construct L x C Matrix	
	Label_Mat = np.zeros((n, C))
	
	for i in xrange(n):
		if Labels[i] != 0:
			Label_Mat[i, :] = -1
			Label_Mat[i, Labels[i] - 1] = 1
	
	return Label_Mat, label_mapping

def label_nodes(G, L, label_mapping):
	"""Labels the nodes in the graph G using data from matrix L.
	   If a node is still unlabeled, label it -1."""
	
	for i in xrange(L.shape[0]):
		i_class = np.nonzero(L[i,:] == 1)[0]
		if i_class.size == 0:
			i_class = (-1, 'no label')
		class_label = int(i_class[0])	

		# Transform labels back to initial numbering:
		label = label_mapping[class_label]
		G.node[i]['label'] = label	
	
	return G

def assess(Graph, Labels_Original, num_changed, preserved_labels, best_guess, label_mapping):
	"""Assess the success of the algorithm by comparing the propagated labels to the original labels.
	   Returns the percentage of nodes labeled correctly."""
	num_nodes = Labels_Original.shape[0]
	num_mislabeled = 0
	
	# Matrix which will be written to file for visualizing errors and unlabeled data
	# 30 = label was preserved. 20 = labeled correctly. 10 = unlabeled. 0 = incorrect
	results = np.zeros(num_nodes)
	for i in xrange(num_nodes):
		NEW, OLD = Graph.node[i]['label'], Labels_Original[i]
		if NEW == OLD: # Labeled Correctly
			results[i] = 20
		elif NEW == -1: # Unlabeled 
			results[i] = 10
			num_mislabeled += 1	
		else:			# Labeled Incorrectly
			best_guess_original = best_guess[i,:]
			best_guess_sorted = sorted(best_guess[i,:], reverse=True)
			
			# Second guess.
			index1 = np.nonzero(best_guess_original == best_guess_sorted[1])[0][0]
			second_guess = label_mapping[index1]
			# Third guess.
			index2 = np.nonzero(best_guess_original == best_guess_sorted[2])[0][0]
			third_guess = label_mapping[index2]
			# Fourth guess.
			index3 = np.nonzero(best_guess_original == best_guess_sorted[3])[0][0]
			fourth_guess = label_mapping[index3]
			# Fifth guess.
			index4 = np.nonzero(best_guess_original == best_guess_sorted[4])[0][0]
			fifth_guess = label_mapping[index4]
			
			if second_guess == OLD:
				results[i] = 16
			elif third_guess == OLD:
				results[i] = 12
			elif fourth_guess == OLD:
				results[i] = 8
			elif fifth_guess == OLD:
				results[i] = 4
			else:
				results[i] = 0
		
			num_mislabeled += 1

	results[preserved_labels == 1] = 30 # Labels preserved
	synopsis = dict([(i, 100*len(np.nonzero(results == 10* i)[0])/(num_nodes+0.0)) for i in xrange(4)])
	
	print "Percentage of nodes whose labels were preserved:", synopsis[3]
	print "Percentage of unlabeled nodes which got properly labeled:", synopsis[2]
#	print "Percentage of unlabeled nodes which got labeled incorrectly:", synopsis[0]
#	print "Percentage of unlabeled nodes which never got labeled:", synopsis[1]
	print 'number of second guesses', len(np.nonzero(results==16)[0])
	print 'number of third guesses', len(np.nonzero(results==12)[0])
	print 'number of fourth guesses', len(np.nonzero(results==8)[0])
	print 'number of fifth guesses', len(np.nonzero(results==4)[0])
	
	print 'In total: {0} percent were labeled correctly, and {1} percent were labeled incorrectly'.format(str(synopsis[3]+synopsis[2]), str(synopsis[1] + synopsis[0]))
	
	##### Construct Heat Map:
	## sort best_guess for each row which you got wrong. 
	## see which index corresponds to the column which has the correct result. 
	## If it is second, label it with 16, if it is third, 12, if fourth, 8, fifth 4. All the rest should stay 0.	
	
	return results
	
def not_converged(y1, y2, tol):
    """Basic convergence check."""
    return np.sum(np.abs(np.asarray(y1 - y2))) > tol
