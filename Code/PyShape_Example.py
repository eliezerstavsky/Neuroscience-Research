file1 = '/home/eliezer/Neuroscience-Research/Label_Prop/testdatalabels.vtk'
file2 = '/home/eliezer/Neuroscience-Research/Label_Prop/testdata_fundi.vtk'

import PyShape

a = PyShape.Shape(id_tag='Label Testing')

a.import_vtk(file1)
a.import_fundi(file2)


