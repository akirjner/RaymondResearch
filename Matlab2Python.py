import os
import pickle
from Structures import Objects
import sys

#Can be replaced in the future with cloud storage, local for now
#############################################################################
all_files = "/Users/andrewkirjner/Desktop/Andrew/RaymondLab/allfiles/"
py_files = "/Users/andrewkirjner/Desktop/Andrew/RaymondLab/pythonfiles/"
##############################################################################


matfiles = sorted(os.listdir(all_files))

for m, matfile in enumerate(matfiles):
	if ".mat" not in matfile: continue
	python_filename = py_files + matfile.replace(".mat", ".pyc")	
	#if os.path.exists(python_filename): continue
	print(matfile)
	file_behavior_data = Objects.BehaviorData(all_files + matfile)
	pickle.dump(file_behavior_data, open(python_filename, 'wb'))



