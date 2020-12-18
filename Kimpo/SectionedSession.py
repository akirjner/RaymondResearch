from collections import namedtuple, OrderedDict
from Kimpo.SaccadeFunctions import *
from Kimpo.DataProcessingHelpers import *


class SectionedSession:
    def __init__(self, cell_session, num_sections, num_bookend_trials):
        self.session = cell_session
        self.section_idxs = get_section_idxs(num_sections)
        self.num_sections = num_sections
        self.num_bookend_trials = num_bookend_trials
        self.cellName = cell_session[GAINS[0]].cell
        self.trialLength = cell_session[GAINS[0]].tlength[0]
        self.conditions = {('x2', 'ipsi') : self.get_sections('x2', 'ipsi'), ('x2', 'contra') : self.get_sections('x2', 'contra'),
                           ('x0', 'ipsi') : self.get_sections('x0', 'ipsi'), ('x0', 'contra') : self.get_sections('x0', 'contra')}

    def get_sections(self, gain, direction):
        gain_session = self.session[gain]
        start = np.array(gain_session.data['contra'])[0]
        trunc_data = get_trunc_data(gain_session, start)
        end_raw = trunc_data['end_time']
        trials = get_trials(start, end_raw)[direction]
        num_trials = len(trials)
        ss = gain_session.sessionraw['ss']
        chunks_eye = chunk_data(trunc_data['evel'])
        chunks_head = chunk_data(trunc_data['hvel'])
        num_samples = chunks_eye.size
        end_trunc = start + num_samples / SAMPLERATE
        chunks_times = np.linspace(start, end_trunc, num_samples).reshape(chunks_eye.shape)
        fr_baselines = get_fr_baselines(ss, trials, num_trials)
        section_evl_array = OrderedDict()
        saccade_bounds = get_saccade_bounds(chunks_eye)
        for s in range(self.num_sections):
            section_evl_array[s] = self.fill_section(s, chunks_eye, chunks_head, chunks_times, ss, fr_baselines, saccade_bounds, direction)
        return section_evl_array

    def fill_section(self, s, chunks_eye, chunks_head, chunks_times, ss, fr_baselines, saccade_bounds, direction):
        ms_chunks_eye = chunks_eye - chunks_eye.mean(axis=1, keepdims=True)
        ms_chunks_head = chunks_head - chunks_head.mean(axis = 1, keepdims = True)
        section_length = int(self.section_idxs['contra'][1])/SAMPLERATE

        idxs = self.section_idxs[direction]
        window = range(idxs[s], idxs[s + 1])
        section_slice = np.s_[idxs[s]:idxs[s + 1]]
        saccades = find_saccades(ms_chunks_eye, saccade_bounds['ubs'], saccade_bounds['lbs'], window)
        saccade_mask = np.ones(chunks_eye.shape[0], dtype=bool)
        saccade_mask[saccades['trials']] = False

        section_eyevel_clean = chunks_eye[saccade_mask, section_slice]
        early_eyevel_mean = np.mean(section_eyevel_clean[0:self.num_bookend_trials, :], axis=(0, 1))
        late_eyevel_mean = np.mean(section_eyevel_clean[-self.num_bookend_trials:, :], axis=(0, 1))

        section_hvel_clean = ms_chunks_head[saccade_mask, section_slice]
        early_hvel_mean = np.mean(section_hvel_clean[0:self.num_bookend_trials, :], axis = (0, 1))
        late_hvel_mean = np.mean(section_hvel_clean[-self.num_bookend_trials:, :], axis= (0, 1))
        section_times_clean = chunks_times[saccade_mask, section_slice]
        early_times_clean = section_times_clean[0:self.num_bookend_trials, :]
        late_times_clean = section_times_clean[-self.num_bookend_trials:, :]
        early_fr_baselines = fr_baselines[saccade_mask][0:self.num_bookend_trials]
        late_fr_baselines = fr_baselines[saccade_mask][-self.num_bookend_trials:]
        early_fr_mean = get_spike_rates(early_times_clean, early_fr_baselines, ss, section_length)
        late_fr_mean = get_spike_rates(late_times_clean, late_fr_baselines, ss, section_length)
        SectionMeans = namedtuple('SectionMeans', ['earlyEyevelMean', 'lateEyevelMean', 'earlyFrMean', 'lateFrMean',
                                                   'earlyHeadMean', 'lateHeadMean'])
        return SectionMeans(early_eyevel_mean, late_eyevel_mean, early_fr_mean, late_fr_mean, early_hvel_mean, late_hvel_mean)