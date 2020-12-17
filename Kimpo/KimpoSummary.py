import sys
from Structures.Objects import SessionInfo
import pickle
import dill
import os
from collections import defaultdict
import os.path
from os import path
from Kimpo.KimpoPlotting import*
from Kimpo.ExcelHelpers import *
from Kimpo.DataProcessingHelpers import *
from Kimpo.SaccadeFunctions import *
from Kimpo.SectionedSession import SectionedSession


sys.path.insert(0, "../")
pd.set_option("display.max_rows", None, "display.max_columns", None)
GAINS = ["x0", "x2"]
TRIAL_LENGTHS = ["250 ms", "500 ms", "1000 ms"]

TRIALVEL = 20
olp_length = 0.1
SAMPLERATE = 2000
olp_contra = np.s_[0:int((olp_length * SAMPLERATE))]
olp_ipsi = np.s_[int(2.192 / 2 * SAMPLERATE):int((2.192 / 2 + olp_length) * SAMPLERATE)]
olp_contra_range = range(int(olp_length * SAMPLERATE))
olp_ipsi_range = range(int(2.192 / 2 * SAMPLERATE), int((2.192 / 2 + olp_length) * SAMPLERATE))
ipsi_start = int(2.192 / 2) * SAMPLERATE


# Can be replaced in the future with cloud storage, local for now
##################################################################################
py_files_path = "/Users/andrewkirjner/Desktop/RaymondDataFiles/pythonfiles/"
#################################################################################

def process_call():
    args = sys.argv
    sheetinfo = get_sheetinfo(args)
    gains = GAINS
    cells = sheetinfo.uniquecells
    sessionmap = get_sessions(sheetinfo, gains, cells)
    return sheetinfo, sessionmap

def create_session_objs(sheetinfo, sessionmap):
    sessionkeys = sessionmap.keys()
    if sheetinfo.sheetname == 'Steps' and path.exists("original_session_objects.pyc"):
        return pickle.load(open("original_session_objects.pyc", "rb"), encoding='latin1')
    elif sheetinfo.sheetname == 'Steps All' and path.exists("allcells_session_objects.pyc"):
        return pickle.load(open("allcells_session_objects.pyc", "rb"), encoding='latin1')
    sessionObjects = []
    for s, sessionkey in enumerate(sessionkeys):
        print(s)
        sessioninfo = sessionmap[sessionkey]
        if not os.path.exists(py_files_path + sessionkey + ".pyc"): continue
        session_data = pickle.load(open(py_files_path + sessionkey + ".pyc", "rb"), encoding='latin1')
        ipsi_trial_starts, contra_trial_starts = get_trial_starts(session_data)
        num_trials = len(contra_trial_starts)
        if len(contra_trial_starts) > len(ipsi_trial_starts):
            num_trials = len(ipsi_trial_starts)
        ss_channel = get_channel(session_data, "ss")
        if ss_channel == -1:
            print("Could Not Find Simple Spike Times")
            return
        ss = ss_channel.data
        eyevel_raw = get_channel(session_data, "hevel").data
        ss_frs = get_channel(session_data, 'ssfrLis').data
        headvel = get_channel(session_data, "hhvel").data
        sessioninfo = sessioninfo + [SAMPLERATE]

        ##GO BACK AND CHANGE THIS
        sessionraw = {}
        sessionraw['eye raw'] = eyevel_raw
        sessionraw['head vel'] = headvel
        sessionraw['all frs'] = ss_frs
        sessionraw['ss'] = ss

        sessiondf = {"ipsi": ipsi_trial_starts[0:num_trials], "contra": contra_trial_starts[0:num_trials]}

        sessionObjects.append(SessionInfo(sessioninfo, sessiondf, sessionraw))
    if sheetinfo.sheetname == 'Steps':
        pickle.dump(sessionObjects, open("original_session_objects.pyc", "wb"))
    else:
        pickle.dump(sessionObjects, open("allcells_session_objects.pyc", "wb"))
    return sessionObjects


def create_cell_sessions(session_objs):
    cell_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for session in session_objs:
        cell_groups[session.cell][session.tlength[0]][session.gain] = session
    cell_sessions = []
    for tlength in TRIAL_LENGTHS:
        for cell in cell_groups:
            if not cell_groups[cell][tlength]:
                continue
            cell_sessions.append(cell_groups[cell][tlength])
    return cell_sessions, cell_groups


def create_early_late_dict(sheetinfo, cell_sessions, num_sections, num_bookend_trials):
    """

    :type num_trials: object
    """
    if sheetinfo.sheetname == 'Steps' and path.exists("original_evl_dict.pyc"):
        return dill.load(open("original_evl_dict.pyc", "rb"))
    elif sheetinfo.sheetname == 'Steps All' and path.exists("allcells_evl_dict.pyc"):
        return dill.load(open("allcells_evl_dict.pyc", "rb"))
    early_late_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
    num_sessions = 0
    for c, cell_session in enumerate(cell_sessions):
        if not cell_session[GAINS[0]] or not cell_session[GAINS[1]] or cell_session[GAINS[0]].tlength[1] == 2 or cell_session[GAINS[0]].cell == 'D30-2':
            continue
        print(c)
        sectionedSession = SectionedSession(cell_session, num_sections, num_bookend_trials)
        name = sectionedSession.cellName
        length = sectionedSession.trialLength
        early_late_dict[name][length]['x2']['ipsi'] = sectionedSession.conditions.gainUpIpsi
        early_late_dict[name][length]['x2']['contra'] = sectionedSession.conditions.gainUpContra
        early_late_dict[name][length]['x0']['ipsi'] = sectionedSession.conditions.gainDownIpsi
        early_late_dict[name][length]['x0']['contra'] = sectionedSession.conditions.gainDownContra
        num_sessions = num_sessions + 1
    early_late_dict['num_sessions'] = num_sessions
    if sheetinfo.sheetname == 'Steps':
        dill.dump(early_late_dict, open("original_evl_dict.pyc", "wb"))
    else:
        dill.dump(early_late_dict, open("allcells_evl_dict.pyc", "wb"))
    return early_late_dict

if __name__ == "__main__":
    sheetinfo, sessionmap = process_call()
    session_objs = create_session_objs(sheetinfo, sessionmap)
    cell_sessions, cell_groups = create_cell_sessions(session_objs)
    num_sections = 4
    num_bookend_trials = 8
    early_late_dict = create_early_late_dict(sheetinfo, cell_sessions, num_sections, num_bookend_trials)
    analyze_sections(early_late_dict, num_sections, num_bookend_trials)

