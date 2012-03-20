import PyShape

shape = PyShape.Shape()
shape.import_vtk('/home/eli/Neuroscience-Research/Data_Hemispheres/KKI2009_15_rh_30.vtk')
shape.fix_triangles()
shape.create_vtk('/home/eli/Desktop/rh_30.vtk')
shape.compute_lbo()
