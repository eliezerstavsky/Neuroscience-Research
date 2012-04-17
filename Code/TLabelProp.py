__author__ = 'eli'

import PyShape as PS

shape = PS.Shape()

shape.import_vtk('/home/eli/Neuroscience-Research/Label_Prop/testdatalabels.vtk')

shape.import_fundi('/home/eli/Neuroscience-Research/Label_Prop/testdata_fundi.vtk')

shape.initialize_labels(keep='fundi')

shape.propagate_labels()

