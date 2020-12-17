import numpy as np
import math
import statsmodels as sm
SAMPLERATE = 2000

### chunk_data ###
#
# Arguments:
#    trunc_data - a signal (np array) of length N (e.g. a raw eye velocity trace)
#
#
#  Returns:
#    chunks - an numpy array of dimension num_chunks x chunksize, m is the
#             number of full cycles and  n is the number of points in a sample

def chunk_data(trunc_data):
    # calculate size of the chunk - 2.192 is the number of seconds per cycle
    chunksize = int(2.192 * SAMPLERATE)

    # calculate num_chunks
    num_chunks = int(math.floor(len(trunc_data) / chunksize))

    chunks = np.zeros((num_chunks, chunksize))
    start_idx = 0
    chunk_counter = 0
    # fill chunks array until last full cycle is added
    while True:
        next_trial_idx = start_idx + chunksize
        end_idx = next_trial_idx
        if end_idx > len(trunc_data):
            break
        chunks[chunk_counter, :] = trunc_data[start_idx:end_idx]
        start_idx = next_trial_idx
        chunk_counter = chunk_counter + 1
    return chunks


### find_saccades
#
# Arguments:
#  mean_sub_chunks_eye - numpy array of the same size as chunks_eye, where each row
#                        has been mean subtracted. This is used to correct for systematic
#                        drift errors on alternating cycles
#
#  upper_bounds - a numpy array of size 'chunksize' containing desaccading upper bounds
#                 from get_saccade_bounds(...)
#
#  lower_bounds - a numpy array of size 'chunksize' containing desaccading lower bounds
#                 from get_saccade_bounds(...)
#
#
# Returns:
#  ipsi_saccade_counter - The number of ipsiversive half-cycles containing a saccade in the
#                         olp
#
#  ipsi_saccade_trials - a list of trial/cycle numbers of ipsiversive half-cycles containing a saccade in
#                        the olp
#
#  contra_saccade_counter - The number of contraversive half-cycles containing a saccade in the
#                           olp
#
#  contra_saccade_trials - a list of trial/cycle numbers of contraversive half-cycles containing a saccade in
#                          the olp

def find_saccades(mean_sub_chunks_eye, upper_bounds, lower_bounds, window):
    saccade_counter = 0
    saccade_trials = []
    x = np.arange(mean_sub_chunks_eye.shape[1]) / SAMPLERATE
    # Check each cycle for an olp saccade
    for m, chunk in enumerate(mean_sub_chunks_eye):

        # Check every sample point in OLP - if any have velocity above
        # corresponding upper bound (or below corresponding lower bound)
        # set appropriate boolean flag below to true
        onset_saccade = False
        for i in window:
            if chunk[i] <= lower_bounds[i] or chunk[i] >= upper_bounds[i]:
                onset_saccade = True
        # Update counters/cycle number lists if necessary
        if onset_saccade:
            saccade_counter = saccade_counter + 1
            saccade_trials.append(m)
    return dict(zip(["count", "trials"],[saccade_counter, saccade_trials]))



### get_saccade_bounds ###
#
# Arguments:
#  chunks_eye - a numpy array with rows that are eye velocity trace cycles
#
# Returns:
#  ubs - a 1D numpy array that has length of the number of points in any cycle.
#        For each sample point in a cycle, there is a corresponding upper bound
#        on eye velocity, above which that sample point is considered to be a
#        saccade in the 'positive' direction
#
#  lbs - a 1D numpy array that has length of the number of points in any cycle.
#        For each sample point in a cycle, there is a corresponding lower bound
#        on eye velocity, below which that sample point is considered to be a
#        saccade in the 'negative' direction
#
#  mean_sub_chunks_eye - a numpy array of the same size as chunks_eye, where each row
#                        has been mean subtracted. This is used to correct for systematic
#                        drift errors on alternating cycles.
def get_saccade_bounds(chunks_eye):
    ubs = np.zeros(chunks_eye.shape[1])
    lbs = np.zeros(chunks_eye.shape[1])
    mean_sub_chunks_eye = chunks_eye - chunks_eye.mean(axis=1, keepdims=True)

    # Desaccading Procedure
    clipped_mask = np.ones(chunks_eye.shape[0], dtype=bool)
    for t in range(chunks_eye.shape[1]):
        # First, take a column of the chunks array to get a distribution for
        # a single time point
        clipped_distribution = mean_sub_chunks_eye[:, t]

        clipped_mask[(clipped_distribution > 40) | (clipped_distribution < -40)] = False
        if sum(clipped_mask) > 5:
            clipped_distribution = clipped_distribution[clipped_mask]
        # Then, clip that distribution to only contain feasible velocities. For this
        # data, feasible values were between -40 and 40
        # clipped_distribution = [val for val in tp_distribution if val > -40 and val < 40]

        # Check to see that the clipped distribution is non-empty. If it is, reset it to be the
        # unprocessed timepoint distribution (may need to revise this later)
        # if len(clipped_distribution) == 0:
        #   clipped_distribution = tp_distribution

        # Compute the Median Absolute Deviation statistic, and use it to set the desaccading bounds
        # with relation to the median. (Also may want to change this later, possibly to a mode-based
        # stat or asymmetric trimmed mean)
        MAD = sm.robust.scale.mad(clipped_distribution)
        median = np.median(clipped_distribution)
        ubs[t] = median + 3.5 * MAD
        lbs[t] = median - 3.5 * MAD
    saccade_bounds_dict = dict(zip(["ubs", "lbs"], [ubs, lbs]))
    return saccade_bounds_dict
