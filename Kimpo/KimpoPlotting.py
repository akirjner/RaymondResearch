import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.patches import ConnectionPatch
from Kimpo.DataAnalysisHelpers import *




section_names = ['0-25 ms', '25-50 ms', '50-75 ms', '75-100 ms']
original_cells = ['D30-2', 'E32-2', 'E41-1', 'D54-1', 'D42-1', 'E43-2', 'E41-2', 'E31-4', 'D71-1', 'D83-1']
DIRECTIONS = ['ipsi', 'contra']
GAINS = ['x2', 'x0']

def format_boxplot_axes(axs, num_sections, num_trials):
    for s in range(num_sections):
        early_axs = axs[s, 0]
        late_axs = axs[s, 1]
        early_axs.get_shared_y_axes().join(early_axs, late_axs)
        early_axs.spines['bottom'].set_visible(False)
        early_axs.axes.get_xaxis().set_visible(False)
        early_axs.spines['top'].set_visible(False)
        early_axs.spines['right'].set_visible(False)
        late_axs.spines['bottom'].set_visible(False)
        late_axs.axes.get_xaxis().set_visible(False)
        late_axs.spines['top'].set_visible(False)
        late_axs.spines['left'].set_visible(False)
        late_axs.yaxis.tick_right()
        early_axs.set_title('Open Loop ' + section_names[s] + ' Early Trials ' + "(" + str(num_trials) + ")")
        late_axs.set_title('Open Loop ' + section_names[s] + ' Late Trials ' + "(" + str(num_trials) + ")")




def plot_boxplots(vectors, axs, s, plot_colors, pos_color, neg_color):
    y_early = vectors[s, 0, :]
    x_early = np.random.normal(1, 0.02, len(y_early))
    y_late = vectors[s, 1, :]
    x_late = np.random.normal(1, 0.02, len(y_late))
    axs_early = axs[s, 0]
    axs_late = axs[s, 1]
    adaptive_changes = 0
    for i in range(len(y_early)):
        if y_early[i] > y_late[i]:
            adaptive_changes = adaptive_changes + 1
    adaptive_change_ratio = np.round(float(adaptive_changes) / len(x_early), 3)

    mean_diff = np.round(np.mean(y_late) - np.mean(y_early), 3)
    median_diff = np.round(np.median(y_late) - np.median(y_early), 3)

    axs_early.boxplot(y_early, showmeans=True,
                      meanprops={"marker": "s", "markerfacecolor": "black", "markeredgecolor": "black"},
                      showfliers=False)
    early_fliers = boxplot_stats(y_early)[0]['fliers']
    axs_late.boxplot(y_late, showmeans=True,
                     meanprops={"marker": "s", "markerfacecolor": "black", "markeredgecolor": "black"},
                     showfliers=False)
    late_fliers = boxplot_stats(y_late)[0]['fliers']

    outlier_idxs = [i for i in range(len(y_late)) if y_late[i] in late_fliers or y_early[i] in early_fliers]
    outlier_mask = np.ones(len(y_late), dtype=bool)
    outlier_mask[outlier_idxs] = 0

    x_early = x_early[outlier_mask]
    y_early = y_early[outlier_mask]
    x_late = x_late[outlier_mask]
    y_late = y_late[outlier_mask]

    plot_colors = np.array(plot_colors)[outlier_mask]
    axs_early.scatter(x_early, y_early, marker='.', c=plot_colors)
    axs_late.scatter(x_late, y_late, marker='.', c=plot_colors)

    xy_early = np.column_stack((x_early, y_early))
    xy_late = np.column_stack((x_late, y_late))

    for j in range(xy_early.shape[0]):
        xy_early_point = xy_early[j, :]
        xy_late_point = xy_late[j, :]
        c = pos_color
        if xy_late_point[1] < xy_early_point[1]:
            c = neg_color
        elif xy_late_point[1] == xy_early_point[1]:
            c = 'black'
        con = ConnectionPatch(xyA=xy_late_point, xyB=xy_early_point, coordsA='data', coordsB='data',
                              axesA=axs_late, axesB=axs_early, linewidth=0.5,
                              linestyle='dotted', color=c)
        axs_late.add_artist(con)

    early_xlim = axs_early.axes.get_xlim()
    early_ylim = axs_late.axes.get_ylim()

    late_xlim = axs_early.axes.get_xlim()
    late_ylim = axs_late.axes.get_ylim()

    xy_top = np.array([[early_xlim[0], early_ylim[1]], [late_xlim[1], late_ylim[1]]])
    xy_bottom = np.array([[early_xlim[0], early_ylim[0]], [late_xlim[1], late_ylim[0]]])
    con_top = ConnectionPatch(xyA=xy_top[1, :], xyB=xy_top[0, :], coordsA='data', coordsB='data', axesA=axs_late,
                              axesB=axs_early, linewidth=0.7)
    con_bottom = ConnectionPatch(xyA=xy_bottom[1, :], xyB=xy_bottom[0, :], coordsA='data', coordsB='data',
                                 axesA=axs_late, axesB=axs_early, linewidth=0.7)

    axs_late.add_artist(con_top)
    axs_late.add_artist(con_bottom)
    axs_early.text(0.2, 0.9, "Adaptive Change \nRatio: " + str(adaptive_change_ratio), ha='center', va='center',
                   color='k',
                   fontsize='medium', fontweight='semibold', transform=axs_early.transAxes,
                   bbox=dict(facecolor='none', edgecolor='k', pad=3))
    axs_early.text(0.2, 0.5, "Mean Difference: \n" + str(mean_diff), ha='center', va='center', color='k',
                   fontsize='medium', fontweight='semibold', transform=axs_early.transAxes,
                   bbox=dict(facecolor='none', edgecolor='k', pad=3))
    axs_early.text(0.21, 0.1, "Median Difference: \n" + str(median_diff), ha='center', va='center', color='k',
                   fontsize='medium', fontweight='semibold', transform=axs_early.transAxes,
                   bbox=dict(facecolor='none', edgecolor='k', pad=3))

def section_boxplots(early_late_dict, num_sections, num_trials):

    num_sessions = early_late_dict['num_sessions']
    cells = [key for key in early_late_dict.keys() if key != 'num_sessions']
    for gain in GAINS:
        for direction in DIRECTIONS:
            pos_color = 'red'
            neg_color = 'darkgreen'
            if (gain == 'x2' and direction == 'contra') or (gain == 'x0' and direction == 'ipsi'):
                pos_color = 'darkgreen'
                neg_color = 'red'
            session_counter = 0
            plot_colors = []
            eye_boxplot_vectors = np.zeros((2, 2, num_sessions))
            fr_boxplot_vectors = np.zeros((2, 2, num_sessions))
            fig_eye, axs_eye = plt.subplots(num_sections, 2, tight_layout=True, figsize = (15, 10))
            fig_eye.suptitle(gain + "-" + direction + " Average Eye Velocity\n (n = " + str(num_sessions) + ")")
            fig_eye.text(0, 0.5, 'deg/s', rotation='vertical', fontsize='large')
            fig_fr, axs_fr = plt.subplots(num_sections, 2, tight_layout=True, figsize = (15, 10))
            fig_fr.suptitle(gain + "-" + direction + " Baseline Subtracted\n Average Firing Rate (n = " + str(num_sessions) + ")")
            fig_fr.text(0, 0.5, 'deg/s', rotation='vertical', fontsize='large')
            format_boxplot_axes(axs_eye, num_sections, num_trials)
            format_boxplot_axes(axs_fr, num_sections, num_trials)
            for cell in cells:
                tlengths = early_late_dict[cell].keys()
                for tlength in tlengths:
                    condition = early_late_dict[cell][tlength][gain][direction]
                    for s in range(num_sections):
                        condition_section = condition[s]
                        eye_boxplot_vectors[s,:,session_counter] = np.array((condition_section.earlyEyevelMean, condition_section.lateEyevelMean))
                        fr_boxplot_vectors[s,:,session_counter] = np.array((condition_section.earlyFrMean, condition_section.lateFrMean))
                    if cell in original_cells:
                        plot_colors.append('blue')
                    else:
                        plot_colors.append('purple')
                    session_counter = session_counter + 1
            for i in range(num_sections):
                plot_boxplots(eye_boxplot_vectors, axs_eye, i, plot_colors, pos_color, neg_color)
                plot_boxplots(fr_boxplot_vectors, axs_fr, i, plot_colors, pos_color, neg_color)
            fig_eye.savefig("./SectionPlots/Boxplots/" + gain + direction + "EyevelBoxplots.png", dpi = 200)
            fig_fr.savefig("./SectionPlots/Boxplots/" + gain + direction + "FrBoxplots.png", dpi = 200)

def plot_eye_vs_fr(eye_data, fr_data, axs, gds, title, pc):
    pos_color = 'red'
    neg_color = 'darkgreen'
    if (gds[0] == 'x2' and gds[1] == 'contra') or (gds[0] == 'x0' and gds[1] == 'ipsi'):
        pos_color = 'darkgreen'
        neg_color = 'red'
    faxs = axs.ravel()
    faxs[pc].grid(True)
    faxs[pc].yaxis.set_ticks_position('left')
    faxs[pc].yaxis.set_tick_params(labelsize=8.5)
    eye_diffs = np.subtract(np.array(tuple(eye_data['late'])),np.array(tuple(eye_data['early'])))
    fr_diffs = np.subtract(np.array(tuple(fr_data['late'])), np.array(tuple(fr_data['early'])))
    diffs_min1, diffs_min2 = np.argpartition(eye_diffs, 1)[0:2]
    diffs_max = np.argmax(eye_diffs)
    eye_diffs = np.delete(eye_diffs, np.array((diffs_min1, diffs_min2, diffs_max)))
    fr_diffs = np.delete(fr_diffs, np.array((diffs_min1, diffs_min2, diffs_max)))

    ylab = 'Change in Firing Rate (sp/s)'
    xlab = 'Change in Eye Velocity (deg/s)'

    x1 = np.arange(0, 5)
    x3 = np.arange(-5, 1)
    faxs[pc].fill_between(x1, 0, 50, color=pos_color, alpha=0.3)
    faxs[pc].fill_between(x3, -50, 0, color=neg_color, alpha=0.3)
    faxs[pc].set_xlim([-4, 4])
    faxs[pc].set_xlabel(xlab)
    faxs[pc].set_ylabel(ylab)
    faxs[pc].set_ylim([-50, 50])
    faxs[pc].plot(range(-100, 100), [0]*200, color = 'k', linewidth = 1.25)
    faxs[pc].plot([0]*200, range(-100, 100), color = 'k', linewidth = 1.25)

    faxs[pc].scatter(eye_diffs, fr_diffs, color = 'k')
    fit, fit_line, mod = bestfit(eye_diffs, fr_diffs)
    faxs[pc].plot(fit_line[0], fit_line[1])
    textstr = '\n'.join((
        r'$\mathrm{slope}=%.2f$' % (fit[1],),
        r'$\mathrm{pval}=%.4f$' % (mod.pvalues[1],),
        r'$\mathrm{R2}=%.2f$' % (mod.rsquared,)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    faxs[pc].text(0.8, 0.95, textstr, transform=faxs[pc].transAxes, fontsize=11,
                 verticalalignment='top', bbox=props)
    faxs[pc].set_title(ylab + ' vs ' + xlab + "\n" + title, loc = 'center')

def section_eye_fr_plots(early_late_dict, num_sections):
    plot_name = 'EyeVsFr.png'
    for gain in GAINS:
        for direction in DIRECTIONS:
            fig, axs = plt.subplots(2, 2, tight_layout=True, figsize=(15, 10))
            fig.suptitle(gain + " " + direction, fontsize = 16)
            pc = 0
            for s in range(num_sections):
                gds = (gain, direction, s)
                eye_data = early_late_dict['eye'][gds]
                fr_data = early_late_dict['fr'][gds]
                plot_eye_vs_fr(eye_data, fr_data, axs, gds, section_names[s], pc)
                pc = pc + 1
            fig.savefig("./SectionPlots/EyeVsFrPlots/" + gain + direction + plot_name, dpi=200)
    for s in range(num_sections):
        fig, axs = plt.subplots(2, 2, tight_layout=True, figsize=(15, 10))
        fig.suptitle(section_names[s], fontsize=16)
        pc = 0
        for gain in GAINS:
            for direction in DIRECTIONS:
                gds = (gain, direction, s)
                eye_data = early_late_dict['eye'][gds]
                fr_data = early_late_dict['fr'][gds]
                plot_eye_vs_fr(eye_data, fr_data, axs, gds, gain + " " + direction, pc)
                pc = pc + 1
        fig.savefig("./SectionPlots/EyeVsFrPlots/" + section_names[s][0:-3] + plot_name, dpi=200)

def get_colors(early_points, late_points, pos_color, neg_color, diff):
    colors = []
    for i in range(len(early_points)):
        if diff:
            if late_points[i] < 0:
                colors.append(neg_color)
            else:
                colors.append(pos_color)
        else:
            if late_points[i] - early_points[i] < 0:
                colors.append(neg_color)
            else:
                colors.append(pos_color)
    return colors

def plot_panel(data, axs, gds, diff, type, title, pc):
    pos_color = 'red'
    neg_color = 'darkgreen'
    if (gds[0] == 'x2' and gds[1] == 'contra') or (gds[0] == 'x0' and gds[1] == 'ipsi'):
        pos_color = 'darkgreen'
        neg_color = 'red'
    faxs = axs.ravel()
    early_points = np.array(tuple(data['early']))
    late_points = np.array(tuple(data['late']))
    xlab = 'Early Eye Velocity' if type == 'eye' else 'Early Firing Rate'
    ylab = 'Late Eye Velocity' if type == 'eye' else 'Late Firing Rate'
    unit = '(deg/s)' if type == 'eye' else '(sp/s)'
    if type == 'eye':
        eye_min1, eye_min2 = np.argpartition(early_points, 1)[0:2]
        eye_max = np.argmax(early_points)
        early_points = np.delete(early_points, np.array((eye_min1, eye_min2, eye_max)))
        late_points = np.delete(late_points, np.array((eye_min1, eye_min2, eye_max)))
    if diff:
        late_points = late_points - early_points
        ylab = ylab.replace('Early ', 'Change in ')
    colors = get_colors(early_points, late_points, pos_color, neg_color, diff)
    axis_range = get_axis_range(type)
    faxs[pc].grid(True)
    faxs[pc].set_title(ylab + ' vs ' + xlab + "\n" + title, loc = 'center', fontsize = 13)
    faxs[pc].set_ylabel(ylab + " " + unit)
    faxs[pc].set_xlabel(xlab + ' ' + unit)
    fit,fit_line,mod = bestfit(early_points, late_points)
    textstr = '\n'.join((
        r'$\mathrm{slope}=%.2f$' % (fit[1],),
        r'$\mathrm{pval}=%.4f$' % (mod.pvalues[1],),
        r'$\mathrm{R2}=%.2f$' % (mod.rsquared,)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    xcoord = 0.8
    ycoord = 0.95
    if fit[1] > 0:
        xcoord = 0.05
    faxs[pc].text(xcoord, ycoord, textstr, transform=faxs[pc].transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    faxs[pc].scatter(early_points, late_points, c = colors)
    faxs[pc].plot(fit_line[0], fit_line[1])
    faxs[pc].plot(range(-100, 100), [0]*200, color = 'k', linewidth = 1.25)
    faxs[pc].plot([0]*200, range(-100, 100), color = 'k', linewidth = 1.25)
    faxs[pc].set_xlim([axis_range[0], axis_range[-1]])
    if diff and type == 'eye':
        faxs[pc].set_ylim([-5, 5])
    else:
        faxs[pc].set_ylim([axis_range[0], axis_range[-1]])
    if not diff:
        xpoints = ypoints = faxs[pc].get_xlim()
        faxs[pc].plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)

def section_scatterplots(early_late_dict, num_sections, diff):
    eye_path = "./SectionPlots/Scatterplots/Eye/LateVsEarly/"
    fr_path = "./SectionPlots/Scatterplots/FR/LateVsEarly/"
    eye_plot_name = 'Eyevel.png'
    fr_plot_name = 'Fr.png'
    if diff:
        eye_path = "./SectionPlots/Scatterplots/Eye/DiffsVsEarly/"
        fr_path = "./SectionPlots/Scatterplots/FR/DiffsVsEarly/"
        eye_plot_name = 'DiffsEyevel.png'
        fr_plot_name = 'DiffsFr.png'
    for gain in GAINS:
        for direction in DIRECTIONS:
            fig_eye_cond, axs_eye_cond = plt.subplots(2, 2, tight_layout=True, figsize = (15, 10))
            fig_eye_cond.suptitle(gain + " " + direction, fontsize = 16)
            fig_fr_cond, axs_fr_cond = plt.subplots(2, 2, tight_layout=True, figsize = (15, 10))
            fig_fr_cond.suptitle(gain + " " + direction, fontsize = 16)
            pc = 0
            for s in range(num_sections):
                gds = (gain, direction, s)
                eye_data = early_late_dict['eye'][gds]
                fr_data = early_late_dict['fr'][gds]
                plot_panel(eye_data, axs_eye_cond, gds, diff, 'eye', section_names[s], pc)
                plot_panel(fr_data, axs_fr_cond, gds, diff, 'fr', section_names[s], pc)
                pc = pc + 1
            fig_eye_cond.savefig(eye_path + gain + direction + eye_plot_name, dpi = 200)
            fig_fr_cond.savefig(fr_path + gain + direction + fr_plot_name, dpi = 200)
    for s in range(num_sections):
        fig_eye_sec, axs_eye_sec = plt.subplots(2, 2, tight_layout=True, figsize=(15, 10))
        fig_eye_sec.suptitle(section_names[s], fontsize=16)
        fig_fr_sec, axs_fr_sec = plt.subplots(2, 2, tight_layout=True, figsize=(15, 10))
        fig_fr_sec.suptitle(section_names[s], fontsize=16)
        pc = 0
        for gain in GAINS:
            for direction in DIRECTIONS:
                gds = (gain, direction, s)
                eye_data = early_late_dict['eye'][gds]
                fr_data = early_late_dict['fr'][gds]
                plot_panel(eye_data, axs_eye_sec, gds, diff, 'eye', gain + " " + direction, pc)
                plot_panel(fr_data, axs_fr_sec, gds, diff, 'fr', gain + " " + direction, pc)
                pc = pc + 1
        fig_eye_sec.savefig(eye_path + section_names[s][0:-3] + eye_plot_name, dpi=200)
        fig_fr_sec.savefig(fr_path + section_names[s][0:-3] + fr_plot_name, dpi=200)
    #plt.show()


def analyze_sections(early_late_dict, num_sections, num_trials):
    #section_boxplots(early_late_dict, num_sections, num_trials)
    diff = False
    section_scatterplots(early_late_dict, num_sections, diff)
    section_eye_fr_plots(early_late_dict, num_sections)



