import sys
sys.path.insert(0, "../")
from enum import IntEnum
import pandas as pd
from Structures.Objects import BehaviorData, Channel, SheetInfo, SessionInfo
from Plotting.plotfrs import plot_firing_rates
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm

NUMCELLS = 5
MONKEYS = ["Darwin", "Edison"]
GAINS = ["x2", "x0"]
TRIAL_LENGTHS = ['250 ms', '500 ms', '1000 ms']


#Can be replaced in the future with cloud storage, local for now
#############################################################################
py_files_path = "/Users/andrewkirjner/Desktop/Andrew/RaymondDataFiles/pythonfiles/"
##############################################################################


class Monkeys(IntEnum):
	BOTH = 0
	DARWIN = 1
	EDISON = 2



	
def get_channel(session_data, chan_name):
	channels = session_data.channels
	for i in range(len(channels)):
		if channels[i].name == chan_name:
			return channels[i]
	return -1

def get_spk_idx(ss_times, time):
	spk_idx = -1
	for idx in reversed(range(len(ss_times))):
		if (time - ss_times[idx]) > 0:
			spk_idx = idx
			return idx
	return spk_idx


def session_names(session_data):
	for channel in session_data.channels:
		print(channel.name)

def get_gains(args):
	gains = [g for g in GAINS if "-" + g in args]
	if len(gains) == 0:
		gains = GAINS
	return gains

def get_lengths(args):
	tlflags = ["-" + l.replace(" ", "") for l in TRIAL_LENGTHS]
	lengths = [TRIAL_LENGTHS[i] for i in range(len(tlflags)) if tlflags[i] in args]
	if len(lengths) == 0:
		lengths = TRIAL_LENGTHS
	return lengths

def get_mid(args):
	mid = 0
	if '-darwin' in args or '-Darwin' in args:
		mid = mid + 1 #Monkeys.Darwin = 1
	if "-edison" in args or '-Edison' in args:
		mid = mid + 2 #Monkeys.Darwin = 2
	return mid % 3

def get_sheetinfo(args):
	sheetname = 'Steps'
	step_lengths = ['250 ms', '500 ms', '1000 ms']
	gains = ['x2', 'x0']
	sheet = pd.read_excel('trialbytrial.xlsx', sheet_name = sheetname)
	if '-allcells'in args or '-oneplotsummaries' in args:	
		sheetname = 'Steps All'
		full_sheet = pd.read_excel('trialbytrial.xlsx', sheet_name = sheetname)
		colstart = full_sheet.columns[0]
		colend = full_sheet.columns[3]
		sheet = full_sheet[(full_sheet['HGVP'] == 1) &
			np.logical_or.reduce([full_sheet['TRAINING'] == g for g in gains]) &
				 np.logical_or.reduce([full_sheet['STEP LENGTH'] == l for l in step_lengths]) &
					(full_sheet['Notes'].str.contains('messy', case = False) != 1)].loc[:,colstart:colend]
	#########
	if '-sines' in args:
		sheetname = 'Sines All'
	sheetinfo = SheetInfo(sheet, sheetname)
	print(sheetinfo.uniquecells)
	return sheetinfo

def get_cells(mid, sheetinfo, args):
	cells = [k for k in sheetinfo.uniquecells if "--" + k in args]
	if len(cells) != 0:
		return cells
	else:
		n = len(sheetinfo.uniquecells)
		if mid == Monkeys.DARWIN:
			cells = sheetinfo.uniquecells[0:n/2]
		elif mid == Monkeys.EDISON:
			cells = sheetinfo.uniquecells[n/2:n]
		else:
			cells = sheetinfo.uniquecells
		return cells

def get_session_list(sheetinfo):
	if os.path.exists(py_files_path + "session_file_list.pyc"):
		return pickle.load(open(py_files_path + "session_file_list.pyc", "rb"))
	session_list = []
	pyfiles = os.listdir(py_files_path)
	for file in pyfiles:
		if ".pyc" not in file: continue
		session_list.append(file.replace(".pyc", ""))
	pickle.dump(session_list, open(py_files_path + "session_file_list.pyc", "wb"))
	return session_list

def get_ssfrlis(ss_channel, sample_rate):
	ss_times = ss_channel.data
	#(ss_times)
	end_time = ss_channel.tend
	times = np.linspace(0, end_time, num = sample_rate*end_time)
	#print(len(ss_times))
	#print(times)
	frs = np.linspace(0.0, 0.0, num = sample_rate*end_time)
	for t in range(times.size):
		spk_indices = np.nonzero(ss_times < times[t])
		if len(spk_indices[0]) == 0: 
			continue
		spk_idx = spk_indices[0][-1]
		#print(spk_idx)
		if spk_idx > 0 and (
			times[t] - ss_times[spk_idx]) < (ss_times[spk_idx] - ss_times[spk_idx-1]):
				frs[t] = float(1)/(ss_times[spk_idx] - ss_times[spk_idx-1])
		elif spk_idx < len(ss_times) - 1:
			frs[t] = float(1)/(ss_times[spk_idx + 1] - ss_times[spk_idx])
	return frs


def add_to_sessionmap(sheetinfo, sessionmap, sessionkey, cell = None, gain = None, tlength = None):
	monkey = 'Darwin' if 'da' in sessionkey else 'Edison'
	if gain == None or tlength == None or cell == None:
		gain = list(sheetinfo.gains[sheetinfo.files == sessionkey])[0]
		tlength = list(sheetinfo.lengths[sheetinfo.files == sessionkey])[0]
		cell = list(sheetinfo.allcells[sheetinfo.files == sessionkey])[0]
	sessionmap[sessionkey] = [monkey, cell, gain, tlength]


def get_sessions(sheetinfo, gains, lengths, cells, args):
	session_list = get_session_list(sheetinfo)
	sessionmap = {}
	
	#Checking if sessions were specified in the arguments
	sessionkeys = [s for s in session_list if "-" + s  in args and s in sheetinfo.uniquecells]	
	if len(sessionkeys) != 0:
		for key in sessionkeys:
			add_to_sessionmap(sheetinfo, sessions, sessionkey)
		return sessions
	#If one of the specified session is not found, return empty session dictionary
	if len(sys.argv) > 1 and any(['.0' in sys.argv]):
		return None
	
	#Otherwise fill session dictionary -- {sessionname: [monkey, cell, gain, length]}
	for c, cell in enumerate(cells):
		print(cell)
		for g, gain in enumerate(gains):
			for l, length in enumerate(lengths):
				sessionkeylist = list(sheetinfo.files[(sheetinfo.gains == gain) & 
				(sheetinfo.lengths == length) & (sheetinfo.allcells == cell)])
				if len(sessionkeylist) == 0: continue
				add_to_sessionmap(sheetinfo, sessionmap, sessionkeylist[0], cell, gain, length)
	return sessionmap


def get_trial_starts(sheetinfo, sessionkey, session_data):
	
	session_gain_list = list(sheetinfo.gains[sheetinfo.files == sessionkey])
	session_gain = session_gain_list[0]
	side = ''
	if session_gain == "x0":
		side = 'contra'
		if "-oppside" in sys.argv:
			side = 'ipsi'
	elif session_gain == "x2":
		side = 'ipsi'
		if "-oppside" in sys.argv:
			side = 'contra'
	trial_starts = [time + 0.004 for time in get_channel(session_data, side).data]
	return np.array(trial_starts)


def get_frs(all_frs, sample_rate, trial_starts):
	frs = []
	for tstart in trial_starts:
		index = int(round(tstart/(1/sample_rate)))
		frs.append(all_frs[index])
	return np.array(frs)

def get_ssfrBase(ss_channel, trial_starts):
	ss_times = ss_channel.data
	baseline_frs = np.linspace(0.0, 0.0, num = len(trial_starts))
	for i in range(len(baseline_frs)):
		baseline_start = 0
		baseline_end = 0
		if i == 0:
			baseline_start = ss_times[i]
			baseline_end = ss_times[i+2]
		elif i == len(baseline_frs)-1:
			baseline_start = ss_times[i-2]
			baseline_end = ss_times[i]
		else:
			baseline_start = ss_times[i-1]
			baseline_end = ss_times[i+1]
		baseline_rate = len(np.nonzero((ss_times >= baseline_start) & (ss_times <= baseline_end)))/(baseline_end - baseline_start)
		baseline_frs[i] = baseline_rate
	return baseline_frs

def create_session_objs(sheetinfo, sessionmap):
	sessionkeys = sessionmap.keys()
	sessionObjects = []
	for s, sessionkey in enumerate(sessionkeys):
		if not os.path.exists(py_files_path + sessionkey + ".pyc"): continue
		session_data = pickle.load(open(py_files_path + sessionkey + ".pyc", "rb"))
		trial_starts = get_trial_starts(sheetinfo, sessionkey, session_data)
		ss_channel = get_channel(session_data, "ss")
		sample_rate = get_channel(session_data, "hevel").samplerate
		if ss_channel == -1:
			print("Could Not Find Simple Spike Times")
			return
		ss_frs = get_frs(get_channel(session_data, "ssfrLis").data, sample_rate, trial_starts)
		ss_frs_baseline = get_ssfrBase(ss_channel, trial_starts)
		m, b = np.polyfit(trial_starts, ss_frs - ss_frs_baseline, 1)
		m_baseline, b_baseline = np.polyfit(trial_starts, ss_frs_baseline,1)
		bestfit = b + m*trial_starts
		bestfit_baseline = b_baseline + m_baseline*trial_starts
		t_starts = sm.add_constant(trial_starts)
		bs_sub = ss_frs - ss_frs_baseline
		bs_sub_mod = sm.OLS(bs_sub, t_starts)
		bs_sub_res = bs_sub_mod.fit()
		p_val_bs_sub = bs_sub_res.pvalues
		t_val_bs_sub = bs_sub_res.tvalues


		sessioninfolist = sessionmap[sessionkey] + [ss_frs, ss_frs_baseline, bestfit, m, trial_starts, p_val_bs_sub[1], t_val_bs_sub[1]]
		sessionObjects.append(SessionInfo(sessioninfolist))
	return sessionObjects

def get_plot_params():
	plot_params = []
	if "-comparegains" in sys.argv:
		plot_params.append("comparegains")
		if len(sys.argv) > 2 and sys.argv.index("-comparegains") != len(sys.argv)-1 and sys.argv[sys.argv.index("-comparegains") + 1] == "-overlay":
			plot_params.append("overlay-gains")
	if "-comparelengths" in sys.argv:
		plot_params.append("comparelengths")
		if len(sys.argv) > 2 and sys.argv.index("-comparelengths") != len(sys.argv)-1 and sys.argv[sys.argv.index("-comparelengths") + 1] == "-overlay":
			plot_params.append("overlay-lengths")
	if "-allconditions" in sys.argv:
		plot_params.append("allconditions")
		if "-overlay-g" in plot_params:
			plot_params.append("overlay-gains")
		if "-overlay-l" in plot_params:
			plot_params.append("overlay-lengths")
	if "-bymonkey" in sys.argv:
		plot_params.append("bymonkey")
	if "-oppside" in sys.argv:
		plot_params.append("oppside")
	if "-oneplotsummaries" in sys.argv:
		plot_params.append("oneplotsummaries")


	return plot_params


def process_call():
	args = sys.argv	
	sheetinfo = get_sheetinfo(args)
	print(sheetinfo.allcells)
	gains = get_gains(args)
	lengths = get_lengths(args)
	mid = get_mid(args)
	cells = get_cells(mid, sheetinfo, args)
	sessionmap = get_sessions(sheetinfo, gains, lengths, cells, args)
	return sheetinfo, sessionmap

if __name__ == "__main__":
	sheetinfo, sessionmap = process_call()
	if session_names == None or len(sessionmap) == 0:
		print("No Sessions Found")
	else:
		session_objs = create_session_objs(sheetinfo, sessionmap)
		plot_params = get_plot_params()
		plot_firing_rates(session_objs, sheetinfo, plot_params)























