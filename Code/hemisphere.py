# Hemisphere Analysis! Study inter-hemispheric asymmetry...

import numpy as np
from time import time
import vtk_operations as vo
import Shape_Analysis as SA
import pyvtk

### Modify to look at all files.

left_file = '/home/eli/lh.aparcNMMjt.inflated.vtk' #KKI2009-15 # OR PIAL
right_file = '/home/eli/rh.aparcNMMjt.inflated.vtk' #KKI2009-15 # OR PIAL

left_data = pyvtk.VtkData(left_file)
right_data = pyvtk.VtkData(right_file)

left_nodes = np.asarray(left_data.structure.points)
left_meshes = np.asarray(left_data.structure.polygons)
left_labels = np.asarray(left_data.point_data.data[0].scalars)

right_nodes = np.asarray(right_data.structure.points)
right_meshes = np.asarray(right_data.structure.polygons)
right_labels = np.asarray(right_data.point_data.data[0].scalars)

left_set = set(left_labels)
right_set = set(right_labels)

# Check that sets are equivalent...

intersection = left_set.intersection(right_set)
# intersection = intersection.difference(set([0])) # Label zero has no meshes for brain 15 lh pial. (Check if correct.)

# Think of all checks...
########################

for label in intersection:
	file_name = '/home/eli/Neuroscience-Research/Inflated_Hemispheres_KKI2009_15/lh_'+str(label)+'.vtk'
	print file_name
	f = vo.write_header(file_name, msg=str(label))
	
	labeled_indexes = np.nonzero(left_labels == label)
	
	node_list = left_nodes[labeled_indexes]
	
	mesh_list = np.array([[0,0,0]])
	ind = labeled_indexes[0]
	for face in left_meshes:
		a,b,c = face[0], face[1], face[2]
		
		if a in ind and b in ind and c in ind:
			new_a = int((ind==a).nonzero()[0])
			new_b = int((ind==b).nonzero()[0])
			new_c = int((ind==c).nonzero()[0])
			new_face = np.array([new_a,new_b,new_c])
			mesh_list = np.append(mesh_list, new_face[np.newaxis,:], axis=0)
	
	max_node = np.amax(mesh_list.ravel())
	print max_node
	mesh_set = set(mesh_list.ravel())
	full_set = set(xrange(max_node))

	f = vo.write_nodes(f.name, node_list)		
	f = vo.write_edges(f.name, mesh_list[1:])

for label in intersection:	
	file_name = '/home/eli/Neuroscience-Research/Inflated_Hemispheres_KKI2009_15/rh_'+str(label)+'.vtk'
	print file_name
	f = vo.write_header(file_name, msg=str(label))
	
	labeled_indexes = np.nonzero(right_labels == label)
	node_list = right_nodes[labeled_indexes]
	
	mesh_list = np.array([[0,0,0]])
	ind = labeled_indexes[0]
	for face in right_meshes:
		a,b,c = face[0], face[1], face[2]

		if a in ind and b in ind and c in ind:
			new_a = int((ind==a).nonzero()[0])
			new_b = int((ind==b).nonzero()[0])
			new_c = int((ind==c).nonzero()[0])
			new_face = np.array([new_a,new_b,new_c])
			mesh_list = np.append(mesh_list, new_face[np.newaxis,:], axis=0)
	
	max_node = np.amax(mesh_list.ravel())
	print max_node
	mesh_set = set(mesh_list.ravel())
	full_set = set(xrange(max_node))

	f = vo.write_nodes(f.name, node_list)
	f = vo.write_edges(f.name, mesh_list[1:])
