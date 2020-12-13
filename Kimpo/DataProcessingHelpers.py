import numpy as np
olp_length = 0.1
SAMPLERATE = 2000
GAINS = ["x0", "x2"]
TRIAL_LENGTHS = ["250 ms", "500 ms", "1000 ms"]

def shifted(chunks_head):
    first_chunk = chunks_head[0, :]
    shifted = False
    for i in range(1, chunks_head.shape[0]):
        comp_chunk = chunks_head[i, :]
        chunk_xcorr = np.correlate(first_chunk, comp_chunk, mode='full')
        max_lag = np.argmax(chunk_xcorr) - len(first_chunk)
        if abs(max_lag) > 5:
            shifted = True
            break
    return shifted

def get_channel(session_data, chan_name):
    channels = session_data.channels
    for i in range(len(channels)):
        if channels[i].name == chan_name:
            return channels[i]
    return -1

def get_trunc_data(session, start_time):
    head_raw = np.array(session.sessionraw['head vel'])
    head_raw = head_raw[~np.isnan(head_raw)]
    start_idx = int(start_time * SAMPLERATE)
    end_time = len(head_raw) / SAMPLERATE
    trunc_head = np.array(session.sessionraw['head vel'][start_idx:])
    trunc_eye = np.array(session.sessionraw['eye raw'][start_idx:])
    trunc_head = trunc_head[~np.isnan(trunc_head)]
    trunc_eye = trunc_eye[~np.isnan(trunc_eye)]
    return dict(zip(['hvel', 'evel', 'end_time'], [trunc_head, trunc_eye, end_time]))

def get_trial_starts(session_data):
    ipsi_trial_starts = [time for time in get_channel(session_data, 'ipsi').data]
    contra_trial_starts = [time for time in get_channel(session_data, 'contra').data]
    return np.array(ipsi_trial_starts), np.array(contra_trial_starts)

def get_section_idxs(num_sections):
    ipsi_correction = int((2.192) / 2 * SAMPLERATE)
    contra_section_idxs = (np.linspace(0, olp_length, num_sections + 1) * SAMPLERATE).astype(np.int)
    ipsi_section_idxs = contra_section_idxs + ipsi_correction
    return dict(zip(['contra', 'ipsi'], [contra_section_idxs, ipsi_section_idxs]))

def get_fr_baselines(ss, trials, num_trials):
    baselines = np.zeros(num_trials)
    for i in range(num_trials):
        num_bs_spikes = sum(1 for s in ss if (s >= trials[i] - 0.3) and (s <= trials[i]))
        baselines[i] = num_bs_spikes / 0.3
    return baselines

def get_spike_rates(times, baselines, ss, section_length):
    spike_rates = np.zeros(times.shape[0])
    for i in range(times.shape[0]):
        spike_rates[i] = sum(1 for s in ss if times[i, 0] <= s <= times[i, -1])/float(section_length)
        spike_rates[i] = spike_rates[i] - baselines[i]
    return np.mean(spike_rates)

def get_trials(start, end):
    contra_trials = []
    ipsi_trials = []
    trial_length = 2.192
    curr = start
    while (curr + trial_length) <= end:
        contra_trials.append(curr)
        ipsi_trials.append(curr + trial_length / 2)
        curr = curr + trial_length
    return dict(zip(['contra', 'ipsi'], [contra_trials, ipsi_trials]))