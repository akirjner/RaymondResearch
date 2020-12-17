import numpy as np
import pandas as pd
from Structures.Objects import SheetInfo
olp_length = 0.1
SAMPLERATE = 2000
GAINS = ["x0", "x2"]
TRIAL_LENGTHS = ["250 ms", "500 ms", "1000 ms"]

def get_sheetinfo(args):
    sheetname = 'Steps All'
    sheet = pd.read_excel('trialbytrial.xlsx', sheet_name=sheetname)
    if '-allcells' in args:
        sheetname = 'Steps All'
        full_sheet = pd.read_excel('trialbytrial.xlsx', sheet_name=sheetname)
        colstart = full_sheet.columns[0]
        colend = full_sheet.columns[3]
        sheet = full_sheet[(full_sheet['HGVP'] == 1) &
                           np.logical_or.reduce([full_sheet['TRAINING'] == g for g in GAINS]) &
                           np.logical_or.reduce([full_sheet['STEP LENGTH'] == l for l in TRIAL_LENGTHS]) &
                           (full_sheet['Notes'].str.contains('messy', case=False) != 1)].loc[:, colstart:colend]
    print(sheetname)
    sheetinfo = SheetInfo(sheet, sheetname)
    print(len(sheetinfo.uniquecells))
    return sheetinfo


def get_sessions(sheetinfo, gains, cells):
    sessionmap = {}
    for c, cell in enumerate(cells):
        print(cell)
        for g, gain in enumerate(gains):
            for l, length in enumerate(TRIAL_LENGTHS):
                sessionkeylist = list(sheetinfo.files[(sheetinfo.gains == gain) &
                                                      (sheetinfo.lengths == length) & (sheetinfo.allcells == cell)])
                if len(sessionkeylist) == 0: continue
                monkey = 'Darwin' if 'da' in sessionkeylist[0] else 'Elvis'
                sessionmap[sessionkeylist[0]] = [monkey, cell, gain, length, sessionkeylist[0]]
    return sessionmap

