import sys
sys.path.insert(0, "../")
from enum import IntEnum
import pandas as pd
from Structures.Objects import BehaviorData, Channel, SheetInfo, SessionInfo
#from Plotting.plotfrs import plot_firing_rates
#from Plotting.plot_eyevels import visualize_eyevels
#from Plotting.desaccade_plotting import desaccade_plots
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm

NUMCELLS = 5
MONKEYS = ["Darwin", "Elvis"]
GAINS = ["x2", "x0"]
TRIAL_LENGTHS = ['250 ms', '500 ms', '1000 ms']


#Can be replaced in the future with cloud storage, local for now
##################################################################################
py_files_path = "/Users/andrewkirjner/Desktop/RaymondDataFiles/pythonfiles/"
#################################################################################


class Monkeys(IntEnum):
    BOTH = 0
    DARWIN = 1
    ELVIS = 2

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
        print(channel.data)

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
    if "-elvis" in args or '-Elvis' in args:
        mid = mid + 2 #Monkeys.Darwin = 2
    return mid % 3

def get_sheetinfo(args):
    sheetname = 'Steps'
    step_lengths = ['250 ms', '500 ms', '1000 ms']
    gains = ['x2', 'x0']
    sheet = pd.read_excel('trialbytrial.xlsx', sheet_name = sheetname)
    if '-allcells' in args:	
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
        elif mid == Monkeys.ELVIS:
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
    monkey = 'Darwin' if 'da' in sessionkey else 'Elvis'
    if gain == None or tlength == None or cell == None:
        gain = list(sheetinfo.gains[sheetinfo.files == sessionkey])[0]
        tlength = list(sheetinfo.lengths[sheetinfo.files == sessionkey])[0]
        cell = list(sheetinfo.allcells[sheetinfo.files == sessionkey])[0]
    sessionmap[sessionkey] = [monkey, cell, gain, tlength, sessionkey]


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
    ipsi_trial_starts = [time for time in get_channel(session_data, 'ipsi').data]
    contra_trial_starts = [time for time in get_channel(session_data, 'contra').data]
    return np.array(ipsi_trial_starts), np.array(contra_trial_starts)


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

def find_runs(x):
    n = x.shape[0]
    loc_run_start = np.empty(n, dtype=bool)
    loc_run_start[0] = True
    np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
    run_starts = np.nonzero(loc_run_start)[0]

    # find run values
    run_values = x[loc_run_start]

    # find run lengths
    run_lengths = np.diff(np.append(run_starts, n))

    return run_values, run_starts, run_lengths
    
def get_olp_mean_hevel(eyevel, trial_starts, sample_rate):
    trial_eyevels = []
    eye_starts = []
    eye_starts_inds = []
    trial_ends= []

    #print(np.diff(trial_starts))
    #print(trial_starts*sample_rate)
    for t, start in enumerate(trial_starts):

        t_start_index = int(start*sample_rate)
        olp_end = int(t_start_index + 0.1*sample_rate)
        eyevel_olp = eyevel[t_start_index:olp_end]
        eyevel_pre_olp = eyevel[int(t_start_index-0.1*sample_rate):t_start_index]
        trial_ends.append(float(olp_end)/sample_rate)		
        if len(eyevel_pre_olp) != 0:
            if -999 in eyevel_pre_olp: continue
        if -999 in eyevel_olp: continue

        trial_eyevels.append(np.mean(eyevel_olp))
        eye_starts.append(start)
        eye_starts_inds.append(t)

    return np.array(trial_eyevels), np.array(eye_starts), np.array(eye_starts_inds), np.array(trial_ends)

def find_saccades(eyevel_raw):
    eyevel = np.copy(eyevel_raw)
    eyevel_new = np.nan_to_num(eyevel, 0)
    eyevel_new[np.abs(eyevel_new) > np.percentile(eyevel_new, 99)] = -999
    return eyevel_new

def create_session_objs(sheetinfo, sessionmap, params):
    sessionkeys = sessionmap.keys()
    sessionObjects = []
    for s, sessionkey in enumerate(sessionkeys):
        print(s)
        sessioninfo = sessionmap[sessionkey]
        sessioninfo = sessioninfo + [2000]
        if not os.path.exists(py_files_path + sessionkey + ".pyc"): continue
        session_data = pickle.load(open(py_files_path + sessionkey + ".pyc", "rb"))
        ipsi_trial_starts, contra_trial_starts = get_trial_starts(sheetinfo, sessionkey, session_data) 
        # trial_starts = [trial_starts_raw[0]]
        # for i in range(1,len(trial_starts_raw)):
        # 	current_start = trial_starts[len(trial_starts)-1]
        # 	trial_starts.ppend(current_start + 2.192)
        #trial_starts = np.array(trial_starts)
        ss_channel = get_channel(session_data, "ss")
        sample_rate = get_channel(session_data, "hevel").samplerate
        if ss_channel == -1: 
            print("Could Not Find Simple Spike Times")
            return
        eyevel_raw = get_channel(session_data, "hevel").data
        eyevel = find_saccades(eyevel_raw)

        
        #print(eyevel)
        trial_starts = contra_trial_starts
        if len(contra_trial_starts) > len(ipsi_trial_starts):
            trial_starts = ipsi_trial_starts
        ss_frs = get_frs(get_channel(session_data, "ssfrLis").data, sample_rate, trial_starts)
        ss_frs_baseline = get_ssfrBase(ss_channel, trial_starts)
        eyeinfo = get_olp_mean_hevel(eyevel, trial_starts, sample_rate)
        trial_eyevels = eyeinfo[0]
        eye_starts = eyeinfo[1]
        print("Num Saccade Trials: ", len(trial_starts)- len(eye_starts))
        eye_starts_inds = eyeinfo[2]
        # for i in range(len(trial_starts)):
        # 	if i not in eye_starts_inds:
        # 		print(sessionmap[sessionkey], "Open Loop Start ", trial_starts[i], "Open Loop End", eyeinfo[3][i])
        # if"-eyevel" in sys.argv and  sessioninfo[1] =='30-2':
        # 	#zerocolors = np.where(eyevel == 0.0, 'red', 'orange') 
        # 	plt.plot(np.divide(np.arange(len(eyevel_raw)),sample_rate),eyevel_raw)
        # 	plt.plot(np.divide(np.arange(len(eyevel_raw)),sample_rate),eyevel)
        # 	plt.figure(num = s)
        # 	plt.title(sessionmap[sessionkey])
        trial_diff_fill = np.array([np.nan for _ in range(len(trial_starts) - len(eye_starts))])

        m_bs_sub, b_bs_sub = np.polyfit(trial_starts, ss_frs - ss_frs_baseline, 1)
        m_baseline, b_baseline = np.polyfit(trial_starts, ss_frs_baseline,1)
        m_eyevel, b_eyevel = np.polyfit(eye_starts, trial_eyevels, 1)
        m_eye_vs_frs, b_eye_vs_frs = np.polyfit((ss_frs - ss_frs_baseline)[eye_starts_inds] , trial_eyevels, 1)

        bestfit_bs_sub = b_bs_sub + m_bs_sub*trial_starts
        bestfit_baseline = b_baseline + m_baseline*trial_starts

        bestfit_eyevel = b_eyevel + m_eyevel*eye_starts
        bestfit_eye_vs_frs = b_eye_vs_frs + m_eye_vs_frs*((ss_frs - ss_frs_baseline)[eye_starts_inds])

        t_starts = sm.add_constant(trial_starts)
        bs_sub = ss_frs - ss_frs_baseline
        bs_sub_mod = sm.OLS(bs_sub, t_starts)
        bs_sub_res = bs_sub_mod.fit()
        pval_bs_sub = bs_sub_res.pvalues[1]


        bs_mod = sm.OLS(ss_frs_baseline, t_starts)
        bs_mod_res = bs_sub_mod.fit()
        pval_bs = bs_mod_res.pvalues[1]

        eye_starts_constant = sm.add_constant(eye_starts)
        eyevel_mod = sm.OLS(trial_eyevels, eye_starts_constant)
        eyevel_res = eyevel_mod.fit()
        pval_eyevel = eyevel_res.pvalues[1]

        bs_sub = sm.add_constant(bs_sub[eye_starts_inds])
        evevel_vs_bssub = sm.OLS(trial_eyevels, bs_sub)
        eyevel_vs_bssub_res = evevel_vs_bssub.fit()

        pval_eye_vs_bssub = eyevel_vs_bssub_res.pvalues[1]
        rsquared_eye_vs_bssub = eyevel_vs_bssub_res.rsquared



        sessionvalues = {'slopes': {'FR': m_bs_sub, 'Baseline' : m_baseline, 'Eye' : m_eyevel, 'Eye vs FR' :m_eye_vs_frs}, 
                        'pvalues' : {'FR': pval_bs_sub, 'Baseline' : pval_bs, 'Eye': pval_eyevel, 'Eye vs FR' :pval_eye_vs_bssub},
                        'rsquared': {"Eye vs FR": rsquared_eye_vs_bssub}, "samplerate": sample_rate}

        
    ##GO BACK AND CHANGE THIS
        sessionraw = {}
        sessionraw['eye desacadde'] = eyevel
        sessionraw['eye raw'] = eyevel_raw
        sessionraw['head vel'] = get_channel(session_data, 'hhvel').data
        sessionraw['all frs'] = get_channel(session_data, "ssfrLis").data
        sessionraw['ss'] = ss_channel
        if len(trial_diff_fill) != 0:
            trial_eyevels = np.concatenate((trial_eyevels, trial_diff_fill))
            eye_starts = np.concatenate((eye_starts, trial_diff_fill))
            bestfit_eyevel =  np.concatenate((bestfit_eyevel, trial_diff_fill))
            bestfit_eye_vs_frs = np.concatenate((bestfit_eye_vs_frs, trial_diff_fill))

        sessiondf = {"ipsi": ipsi_trial_starts[0:len(trial_starts)], "contra": contra_trial_starts[0:len(trial_starts)], "Eye Starts": eye_starts , "Firing Rate" : ss_frs, "Baseline Firing Rate" : ss_frs_baseline,
                        "Eye Velocity":  trial_eyevels , "Eye Vel Best Fit Line": bestfit_eyevel, "FR Best Fit Line" : 
                        bestfit_bs_sub, "Baseline Best Fit Line": bestfit_baseline, "Eye vs FR Best Fit Line": bestfit_eye_vs_frs }
     

        sessionObjects.append(SessionInfo(sessioninfo, sessiondf, sessionraw))
    return sessionObjects

def get_params():
    params = []
    if "-comparegains" in sys.argv:
        params.append("comparegains")
        if len(sys.argv) > 2 and sys.argv.index("-comparegains") != len(sys.argv)-1 and sys.argv[sys.argv.index("-comparegains") + 1] == "-overlay":
            params.append("overlay-gains")
    if "-comparelengths" in sys.argv:
        params.append("comparelengths")
        if len(sys.argv) > 2 and sys.argv.index("-comparelengths") != len(sys.argv)-1 and sys.argv[sys.argv.index("-comparelengths") + 1] == "-overlay":
            params.append("overlay-lengths")
    if "-allconditions" in sys.argv:
        params.append("allconditions")
        if "-overlay-g" in params:
            params.append("overlay-gains")
        if "-overlay-l" in params:
            params.append("overlay-lengths")
    if "-bymonkey" in sys.argv:
        params.append("bymonkey")
    if "-oppside" in sys.argv:
        params.append("oppside")
    if "-oneplotsummaries" in sys.argv:
        params.append("oneplotsummaries")
    if "-baselines" in sys.argv:
        params.append("baseline")
    if "-eyevel" in sys.argv:
        params.append("eyevel")
    if "-raw" in sys.argv:
        params.append("raw")


    return params


def process_call():
    args = sys.argv	
    sheetinfo = get_sheetinfo(args)
    gains = get_gains(args)
    lengths = get_lengths(args)
    mid = get_mid(args)
    cells = get_cells(mid, sheetinfo, args)
    sessionmap = get_sessions(sheetinfo, gains, lengths, cells, args)
    return sheetinfo, sessionmap

if __name__ == "__main__":
    sheetinfo, sessionmap = process_call()
    plt.close('all')
    if session_names == None or len(sessionmap) == 0:
        print("No Sessions Found")
    else:
        params = get_params()
        session_objs = create_session_objs(sheetinfo, sessionmap, params)
        if "eyevel" in params:

            visualize_eyevels(sheetinfo, session_objs, params)
        # if "raw" in params:
        # 	for s, session in enumerate(session_objs):
        # 		if session.cell == "D30-2":
        # 			print("Hi")
        # 			plt.figure(num = s)
        # 			print(session.sessioneye)
        # 			plt.plot(session.sessioneye)
        # 			plt.title(session.cell + " " + session.tlength[0] + " " + session.gain + " " + session.file)
        # 	plt.show()
        else:
            #plot_firing_rates(session_objs, sheetinfo, params)
            desaccade_plots(session_objs, sheetinfo)























