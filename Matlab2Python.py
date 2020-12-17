import os
import pickle
from Structures import Objects
import sys

#Can be replaced in the future with cloud storage, local for now
#############################################################################
mat_files_path = "/Users/andrewkirjner/Desktop/RaymondDataFiles/matlabfiles/"
py_files_path = "/Users/andrewkirjner/Desktop/RaymondDataFiles/pythonfiles/"
##############################################################################


matfiles = sorted(os.listdir(mat_files_path))

for m, matfile in enumerate(matfiles):
	if ".mat" not in matfile: continue
	python_filename = py_files_path + matfile.replace(".mat", ".pyc")	
	#if os.path.exists(python_filename): continue
	print(matfile)
	file_behavior_data = Objects.BehaviorData(mat_files_path + matfile)
	pickle.dump(file_behavior_data, open(python_filename, 'wb'))



