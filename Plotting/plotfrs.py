import sys
sys.path.insert(0, "../")
from Structures.Objects import SessionInfo
import pandas as pd
import numpy as np
import chart_studio.plotly as py
import plotly.graph_objects as go
from plotly import subplots
import plot_utils as pltutils

GAINS = ['x2', 'x0']
GAIN_NAMES = [' Gain Up (x2) Ipsi', ' Gain Down (x0) Contra']
GAIN_NAMES_OPP = [' Gain Up (x2) Contra', ' Gain Down (x0) Ipsi']

COLORS = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']
LENGTHS = [' 250 ms', ' 500 ms', ' 1000 ms']
SYMBOLS = ['circle', 'square', 'triangle-up']


##One Plot Summary Trace 
def make_one_plot_traces(fig, sessions, sheetinfo, key):
	cellsinfo = pltutils.createcellsinfo(sessions, sheetinfo, 'FR')
	cellsinfo_baseline = pltutils.createcellsinfo(sessions, sheetinfo, 'Baseline')
	y_up = [[], [], []]
	y_up_baseline = [[],[],[]]
	y_down = [[],[],[]]
	y_down_baseline = [[],[],[]]
	x_up = sorted(cellsinfo.keys())
	x_down = sorted(cellsinfo.keys())
	gains = GAINS
	tlengths = LENGTHS
	sel_up = []
	for c, cell in enumerate(x_up):
		for t, tlength in enumerate(tlengths):
			slope = cellsinfo[cell]['x2'][t]
			y_up[t].append(slope)

			slope_baseline = cellsinfo_baseline[cell]['x2'][t]
			y_up_baseline[t].append(slope_baseline)
	for cell in x_down:
		for t, tlength in enumerate(tlengths):
			slope = cellsinfo[cell]['x0'][t]
			y_down[t].append(slope)
			
			slope_baseline = cellsinfo_baseline[cell]['x0'][t]
			y_down_baseline[t].append(slope_baseline)
	for ind, y_vals in enumerate(y_up):
		fig.add_trace(go.Scatter(
			x = x_up,
			y = y_up[ind],
			marker = dict(color = COLORS[ind], size = 8.5, symbol = SYMBOLS[ind]),
			mode = 'markers',
			name = LENGTHS[ind] + ' x2')
		)
		fig.add_trace(go.Scatter(
			x = x_up,
			y = y_up_baseline[ind],
			marker = dict(color = COLORS[ind + 3], size = 8.5, symbol = SYMBOLS[ind]),
			mode = 'markers',
			name = LENGTHS[ind]+ ' x2 Baseline')
		)
	for ind, y_vals in enumerate(y_down):
		fig.add_trace(go.Scatter(
			x = x_down,
			y = y_down[ind],
			marker = dict(color = COLORS[ind], size = 8.5, symbol = SYMBOLS[ind]),
			mode = 'markers',
			name = LENGTHS[ind]+ ' x0')
		)
		fig.add_trace(go.Scatter(
			x = x_down,
			y = y_down_baseline[ind],
			marker = dict(color = COLORS[ind+3], size = 8.5, symbol = SYMBOLS[ind]),
			mode = 'markers',
			name = LENGTHS[ind]+ ' x0 Baseline')
		)


#Boxplot Trace
def make_average_trace(fig, average_plot, plot_params):
	for rc, (row, col) in enumerate(average_plot.keys()):
		cell_names = []
		cell_fr_slopes = []
		cell_eye_slopes = []
		pos_sig_pvalues = []
		neg_sig_pvalues = []
		for t, trace_value in enumerate(average_plot[(row, col)]):
			textval = ''
			if "allconditions" in plot_params:
				textval = trace_value[0]
			elif "comparegains" in plot_params:
				textval = trace_value[0] + ", " + trace_value[1]
			elif "comparelengths" in plot_params:
				textval = trace_value[0] + ", " + trace_value[2]
			if "eyevel" not in plot_params:
				textval = textval + ", p-value: " + str(round(trace_value[4], 4))
				if trace_value[4] < 0.05:
					neg_sig_pvalues.append(t)
			else:
				cell_eye_slopes.append(trace_value[4])
			cell_fr_slopes.append(trace_value[3])
			cell_names.append(textval)
		if "eyevel" not in plot_params:
			box = go.Box(
				name = 'Summary Plot',
				y = cell_fr_slopes,
				text = cell_names,
				boxpoints = 'all',
				selectedpoints = neg_sig_pvalues,
				selected = dict(
						   marker = dict(
									color = 'black',
									size = 7 
									)
							    ),
				pointpos = -1.8,
				jitter = 0.25,
				boxmean = 'sd',
				hoverinfo = "text + y",
				showlegend = False
				)
			fig.append_trace(box, row = row, col = col)
		
		else:
			# l3 = zip(cell_fr_slopes, cell_eye_slopes)
			# l3 = sorted(l3)
			# cell_fr_slopes = []
			# cell_eye_slopes = []
			# for l in l3:
			# 	cell_fr_slopes.append(l[0])
			# 	cell_eye_slopes.append(l[1])


			# scatter = go.Scatter(
			# 	name = "Eye Velocity vs. PC Baseline Subtracted Firing Rate",
			# 	x = cell_fr_slopes,
			# 	y = cell_eye_slopes,
			# 	text = cell_names,
			# 	mode = 'lines + markers', 
			# 	opacity = 0.5, 
			# 	marker = dict(symbol = 'circle', opacity = 0.75), 
			# 	line = dict(dash = 'dot', color = COLORS[rc]),
			# 	showlegend = False,
			# 	hoverinfo = 'text+y')
			# fig.append_trace(scatter, row, col)
			# m_eye_fr, b_eye_fr = np.polyfit(np.array(cell_fr_slopes), np.array(cell_eye_slopes), 1)
			# bestfit_eye_fr = m_eye_fr*np.array(cell_fr_slopes) + b_eye_fr
			# line = go.Scatter(
			# 	x = np.array(cell_fr_slopes),
	 	# 		y = bestfit_eye_fr, 
	 	# 		mode = 'lines', 
	 	# 		line = dict(color = COLORS[rc] ), 
	 	# 		name =  "m " +  str(round(m_eye_fr, 4)),
	 	# 		hoverinfo = 'name',
	 	# 		hoverlabel = dict(namelength = -1),
	 	# 		showlegend = False
			# 	)
			# fig.append_trace(line, row, col)
			box = go.Box(
				name = 'Summary Plot',
				y = cell_eye_slopes,
				text = cell_names,
				boxpoints = 'all',
				selectedpoints = neg_sig_pvalues,
				selected = dict(
						   marker = dict(
									color = 'black',
									size = 7 
									)
							    ),
				pointpos = -1.8,
				jitter = 0.25,
				boxmean = 'sd',
				hoverinfo = "text + y",
				showlegend = False
				)
			fig.append_trace(box, row = row, col = col)


	

def add_traces(fig,sessions, sheetinfo, plot_params):
	trace_names = []
	traces = []
	key = 'FR' if 'baseline' not in plot_params else "Baseline"
	if "oneplotsummaries" in plot_params: 
		make_one_plot_traces(fig, sessions, sheetinfo, key)
		
		trace_names = ['Up', 'Up Baseline']*3 + ['Down', 'Down Baseline']*3 
	else:
		average_plot = {}
		for session in sessions:

			plot_attributes = pltutils.get_plot_attributes(plot_params, session.tlength[0], session.gain)
			pltutils.create_scatter_trace(fig, session, plot_params, plot_attributes)
			pltutils.create_line_trace(fig,session, plot_params, plot_attributes)
			trace_name = (session.cell, session.tlength[0], session.gain)
			trace_names = trace_names + [trace_name, trace_name]
			# if "eyevel" in plot_params:
			# 	trace_names + [trace_name, trace_name]


			trace_value = trace_name + (session.slopes[key],session.pvalues[key])
			if "eyevel" in plot_params:
				trace_value = trace_name + (session.slopes['FR'], session.slopes['Eye'])
			plot_row = plot_attributes['row']
			plot_col = plot_attributes['col']
			if (plot_row, plot_col) in average_plot:
				average_plot[(plot_row, plot_col)].append(trace_value)
			else:
				average_plot[(plot_row, plot_col)] = [trace_value]
		make_average_trace(fig, average_plot, plot_params)
	return trace_names


def get_plot_names(name, plot_params, average_plot_names):
	if average_plot_names != None:
		return average_plot_names
	gain_names = GAIN_NAMES if "oppside" in plot_params else GAIN_NAMES_OPP
	length_names = LENGTHS
	if "allconditions" in plot_params:
		return [name + length_names[j] + gain_names[i] for i in range(len(gain_names)) for j in range(len(length_names))]
	elif "comparelengths" in plot_params:
		return length_names
	elif "comparegains" in plot_params:
		return gain_names

def get_annotations(fig, name, sheetinfo, plot_params, average_plot_names = None):
	annotations = []
	plot_names = get_plot_names(name, plot_params, average_plot_names)
	for n, name in enumerate(plot_names):
		title_xcoord = (fig['layout']['xaxis' + str(n+1)]['domain'][0] + fig['layout']['xaxis' + str(n+1)]['domain'][1])/2
		title_ycoord = fig['layout']['yaxis' + str(n+1)]['domain'][1]
		note = dict(
				font = dict(size = 14), 
				xref = "paper", 
				yref = "paper", 
				xanchor = "center", 
				yanchor = "bottom",
				x = title_xcoord,
				y = title_ycoord,
				text = name, 
				showarrow = False)
		annotations.append(note)
	return annotations

def add_average_button(annotations, trace_names, plot_params, average_plot_names):
	print(average_plot_names)
	visibility = [False for _ in range(len(trace_names))] + [True for _ in range(len(average_plot_names))] #Fix Magic Number
	
	label = "Summary Box Plot"
	#if "eyevel" in plot_params:
	#	label = "Eye Velocity vs. PC Baseline Subtracted Firing Rate"
	#	visibility = [False for _ in range(len(trace_names))] + [True for _ in 2*range(len(average_plot_names))]
	average_button = pltutils.create_button(label, visibility, annotations, None)
	return average_button

def add_oneplot_buttons(fig, dropdown_names, trace_names):
	buttons = []
	print(trace_names)
	for n, name in enumerate(dropdown_names):
		visibility = []
		key = 'FR' if 'Baseline' not in name else 'Baseline'
		if n == 0:
			visibility = [True, False, True, False, True, False] + [False for _ in range(6)]
		elif n == 1:
			visibility = [False for _ in range(6)] + [True, False, True, False, True, False] 
		elif n == 2:
			visibility = [False, True, False, True, False, True] + [False for _ in range(6)]
		elif n == 3:
			visibility = [False for _ in range(6)] + [False, True, False, True, False, True]
		elif n == 4:
			visibility = [True, True] + [False for _ in range(4)] + [False for _ in range(6)] 
		elif n == 5:
			visibility = [False, False] + [True, True]  + [False, False]+ [False for _ in range(6)]
		elif n == 6:
			visibility = [False for _ in range(4)] + [True, True] + [False for _ in range(6)]
		elif n == 7:
			visibility = [False for _ in range(6)] + [True, True] + [False for _ in range(4)] 
		elif n == 8:
			visibility = [False for _ in range(6)] + [False, False] + [True, True]  + [False, False]
		elif n == 9: 
			visibility = [False for _ in range(6)] + [False for _ in range(4)] + [True, True]
		# if key == 'Baseline':
		# 	if n == 0:
		# 		visibility = [True if trace == "Up" else False for trace in trace_names]
		# 	else:
		# 		visibility = [False if trace == "Up" else True for trace in trace_names]
		# else:
		# 	if n == 2:
		# 		visibility = [True if trace == "Up Baseline" else False for trace in trace_names]
		# 	else:
		# 		visibility = [False if trace == "Up Baseline" else True for trace in trace_names]
		button =  pltutils.create_button(name, visibility, None, "NoTitle")
		buttons.append(button)
	return buttons

def add_buttons(fig, dropdown_names, trace_names, plot_params, sheetinfo, average_plot_names):
	if "oneplotsummaries" in plot_params:
		return add_oneplot_buttons(fig, dropdown_names, trace_names)
	buttons = []
	for n, name in enumerate(dropdown_names):
		label = ''
		visibility = []
		if len(plot_params) == 0:
			visibility = [True if (name[0] == trace_names[i][0] and name[1] == trace_names[i][1] and name[2] == trace_names[i][2]) else False for i in range(len(trace_names))]
			label = name[0] + " (" + name[1] + ", " + name[2] + ")"
		elif "allconditions" in plot_params:
			visibility = [True if name == trace_names[i][0] else False for i in range(len(trace_names))]
			label = name
		elif "comparelengths" in plot_params:
			visibility = [True if (name[0] == trace_names[i][0] and name[1] == trace_names[i][2]) else False for i in range(len(trace_names))]
			label = name[0] + " (" + name[1] + ")"	
		elif "comparegains" in plot_params:
			visibility = [True if (name[0] == trace_names[i][0] and name[1] == trace_names[i][1]) else False for i in range(len(trace_names))]
			label = name[0] + " (" + name[1] + ")"	
		if average_plot_names != None:
			visibility = visibility + [False for _ in range(len(average_plot_names))]
			if "eyevel" in plot_params:
				visibility = visibility + [False for _ in range(len(average_plot_names))]

		annotations = get_annotations(fig, name, sheetinfo, plot_params)
		button = pltutils.create_button(label, visibility, annotations, plot_params)
		buttons.append(button)
	average_annotations = get_annotations(fig, "average", sheetinfo, plot_params, average_plot_names)
	average_button = add_average_button(average_annotations, trace_names, plot_params, average_plot_names)
	buttons.insert(0,average_button)
	return buttons

def get_names(sessions, plot_params, sheetinfo):
	dropdown_individual = []
	dropdown_average = []
	filename = ""
	if len(plot_params) == 0:
		dropdown_individual = [[session.cell, session.tlength[0], session.gain] for session in sessions]
		filename = "default"
	elif "oneplotsummaries" in plot_params:
		gain_up_name = "Cells Gain Up" + (" (Ipsi)" if "oppside" not in plot_params else " (Contra)")
		gain_down_name = "Cells Gain Down" + (" (Contra)" if "oppside" not in plot_params else " (Ipsi)")
		dropdown_individual.append(gain_up_name)
		dropdown_individual.append(gain_down_name)
		dropdown_individual.append(gain_up_name + " Baseline")
		dropdown_individual.append(gain_down_name + " Baseline")
		dropdown_individual.append(gain_up_name + " 250 ms")
		dropdown_individual.append(gain_up_name + " 500 ms")
		dropdown_individual.append(gain_up_name + " 1000 ms")
		dropdown_individual.append(gain_down_name + " 250 ms")
		dropdown_individual.append(gain_down_name + " 500 ms")
		dropdown_individual.append(gain_down_name + " 1000 ms")
		filename = "one-plot-summaries"
	else:
		gain_names = GAIN_NAMES
		length_names = LENGTHS
		if "allconditions" in plot_params:
			if "oppside" in plot_params:
				gain_names = GAIN_NAMES_OPP
				filename = "oppside-"
			dropdown_average = [length_names[j] + " " + gain_names[i] for i in range(len(gain_names)) for j in range(len(length_names))]
			dropdown_individual = sorted(list(set([session.cell for session in sessions])))
			filename = filename + "allconditions-by-cell"
		elif "comparelengths" in plot_params and "comparegains" in plot_params:
				dropdown_individual = sorted(unique([session.cell for session in sessions]))
				filename = "compare-lengths-and-gains-by-cell"
		elif "comparelengths" in plot_params:
			dropdown_individual = list(set([(session.cell, session.gain) for session in sessions]))
			filename = "compare-lengths-by-cell"
			dropdown_average = length_names
			dropdown_individual.sort(key = lambda x: (x[0], -int(x[1][1])))
		elif "comparegains" in plot_params:
			dropdown_average = gain_names
			dropdown_individual = list(set([(session.cell, session.tlength[0], session.tlength[1]) for session in sessions]))	
			filename = "compare-gains-by-cell"
			dropdown_individual.sort(key = lambda x: (x[0], x[2]))
			dropdown_individual = [(name[0], name[1]) for name in dropdown_individual]
	filename = filename + "-" + sheetinfo.sheetname.replace(" ", '-')
	if "baseline" in plot_params:
		filename = filename + "-baseline"
	if "eyevel" in plot_params:
		filename = filename + "-eyevel" 
	return dropdown_individual, dropdown_average, filename

def set_fig_dimensions(filename,  plot_params, y_ax_title, x_ax_title):
	fig = go.Figure()
	numrows = 1
	numcols = 1
	plot_height = 650
	plot_width = 1605
	if filename == "default" or filename == "one-plot-summaries":
		return fig, plot_height, 1605
	if "allconditions" in filename:
		numrows = 2
		numcols = 3
		plot_height = 950
	if "compare-lengths" in filename:
	 	numrows = 1
	 	numcols = 3
	elif "compare-gains" in filename:
		numrows = 1
		numcols = 2
	fig = subplots.make_subplots(rows = numrows, 
								 cols = numcols, 
								 horizontal_spacing = 0.025
								 )
	if numrows != 1:
		fig.update_xaxes(row = numrows, title_text = x_ax_title)
	else: 
		fig.update_xaxes(title_text = x_ax_title)
	fig.update_xaxes(tickmode = 'auto', range = [0, 90])

	if numcols != 1:
		fig.update_yaxes(col = 1, title_text = y_ax_title)
	else:
		fig.update_yaxes(title_text = y_ax_title)
	fig.update_yaxes(tickmode = 'auto', range = [-100, 280])

	return fig, plot_height, plot_width


def plot_firing_rates(sessions, sheetinfo, plot_params):
	print("Number of Data Points: ", len(sessions))
	sessions.sort(key = lambda s: (s.cell, -int(s.gain[1]), s.tlength[1]))
	dropdown_names, average_plot_names, filename = get_names(sessions, plot_params, sheetinfo)
	y_ax_title = "Baseline Subtracted PC Firing Rate (sp/s)"
	x_ax_title = "Time (seconds)"
	if "baseline" in plot_params:
		y_ax_title = "Baseline PC Firing Rate (sp/s)"
	elif "oneplotsummaries" in plot_params:
		y_ax_title = "Change in PC Firing Rate (Delta sp/s)"
		x_ax_title = "Cell"
	elif "eyevel" in plot_params:
		y_ax_title = "Average Open Loop Eye Velocity (deg/s)"

	fig, plot_height, plot_width = set_fig_dimensions(filename,  plot_params, y_ax_title, x_ax_title)
	trace_names = add_traces(fig, sessions, sheetinfo,  plot_params)
	buttons = add_buttons(fig, dropdown_names, trace_names, plot_params, sheetinfo, average_plot_names)
	button_layer_1_height = 1.10


	fig.update_layout(
		autosize = False,
		height = plot_height,
		width = plot_width,
		margin = go.layout.Margin(l = 10, r = 10, b = 10, t = 15, pad = 3),
		yaxis_title = y_ax_title,
		legend = dict(
			x = 1.005, 
			y = 0.5,
			font = dict(size = 13)
		),
		updatemenus =[
			dict(
				buttons = buttons,
            	direction ="down",
            	pad={"r": 10, "t": 7},
            	showactive=True,
            	x=0.045,
            	xanchor="right",
            	y=button_layer_1_height,
            	yanchor="top"
				)
			],
		)

	py.plot(fig, filename = filename)












