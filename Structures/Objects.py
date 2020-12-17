#Temporary!
#from matloader.mio import loadmat
import sys
import numpy as np
import pandas as pd


sys.path.insert(0, "../")



class SessionResults:
    def __init__(self, results):
        self.cell = results['cell']
        self.tlength = results['tlength']
        self.beta = results['beta']
        self.bfl = results['bfl']
        self.pval = results['pvalue']
        self.r2 = results['r2']
class SheetInfo():
    def __init__(self, sheet, name):
        self.sheetname = name
        self.allcells = sheet['Cell code']
        self.uniquecells = pd.unique(sheet['Cell code'])
        self.files = sheet['FILE NAME']
        self.gains = sheet['TRAINING']
        self.lengths = sheet['STEP LENGTH']

class SessionInfo():
    def __init__(self, sessioninfo, sessiondata, sessionraw):
        self.monkey = sessioninfo[0]
        self.cell = sessioninfo[1] if self.monkey == "Elvis" else self.monkey[0] + sessioninfo[1]
        self.gain = sessioninfo[2] 
        self.tlength = (sessioninfo[3], 0)
        if sessioninfo[3] == '500 ms':
            self.tlength = (sessioninfo[3], 1)
        elif sessioninfo[3] == '1000 ms':
            self.tlength = (sessioninfo[3], 2) 
        self.file = sessioninfo[4]
        self.samplerate = sessioninfo[5]
        self.sessionraw = sessionraw
        self.data = pd.DataFrame(data = sessiondata, index = None)

class Channel:
    def __init__(self, channel_info):
        self.name = "None"
        if len(channel_info[3]) != 0:
            self.name = str(channel_info[3][0])
        if self.name != "None":
            self.tend = channel_info[4][0][0]
            self.tstart = channel_info[5][0][0]
            self.units = "None"
            if len(channel_info[6]) != 0:
                self.units = str(channel_info[6][0])
            self.samplerate = channel_info[7][0]
            if self.name != 'cs' and self.name != 'ss':
                self.samplerate = self.samplerate[0]
            self.data = "None"
            if len(channel_info[8]) != 0:
                self.data = self.get_data(channel_info)


    def get_data(self, channel_info):
        data = []
        if len(channel_info[8][0]) > 1:
            data = channel_info[8][0]
            return data
        else:
            for i in range(len(channel_info[8])):
                data.append(channel_info[8][i][0])
            return np.array(data)


class BehaviorData:
    def __init__(self, filename):
        all_channels_info = loadmat(filename)['beh'][0]
        self.channels = []
        for i in range(len(all_channels_info)):
            channel_info = all_channels_info[i]
            channel = Channel(channel_info)
            if channel.name == "None": continue
            self.channels.append(channel)




