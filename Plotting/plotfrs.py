import sys
sys.path.insert(0, "../")
from Structures.Objects import SessionInfo
import pandas as pd
import numpy as np
import chart_studio.plotly as py
import plotly.graph_objects as go
from plotly import subplots


def get_plot_attributes(trace_name, plot_params):
	color = ''
	symbol = ''
	label = ''
	legend_group = ''
	if "allconditions" in plot_params:
		if trace_name[1] == '250 ms':
			if trace_name[2] == 'x2':
				color = '#1b9e77'
			elif trace_name[2] == 'x0':
				color = '#d95f02'
			symbol = 'circle'
		elif trace_name[1] == '500 ms':
			if trace_name[2] == 'x2':
				color = '#7570b3'
			elif trace_name[2] == 'x0':
				color = '#e7298a'
			symbol = 'triangle-up'
		elif trace_name[1] == '1000 ms':
			if trace_name[2] == 'x2':
				color = '#66a61e'
			elif trace_name[2] == 'x0':
				color = '#e6ab02'
			symbol = 'cross'
	if "comparelengths" in plot_params:
		color = '#1b9e77'
		symbol = 'circle'
		legend_group = 'group1'
		if trace_name[1] == '500 ms':
			color = '#d95f02'
			symbol = 'triangle-up'
			legend_group = 'group2'
		elif trace_name[1] == '1000 ms':
			color = '#7570b3'
			symbol = 'cross'
			legend_group = 'group3'
		label = trace_name[1]
	elif "comparegains" in plot_params:
		color = '#1b9e77'
		symbol = 'circle'
		legend_group = 'group1'
		label = trace_name[2] + (" (ipsi)" if "oppside" not in plot_params else " (contra)")
		if trace_name[2] == 'x0':
			color = '#7570b3'
			symbol = 'cross'
			legend_group = 'group2'
			label = trace_name[2] + (" (contra)" if "oppside" not in plot_params else " (ipsi)")
	return color, symbol, label

def createcellsinfo(sessions, sheetinfo):
	cellsinfo = {}
	for cell in sheetinfo.uniquecells:
		if cell[0] != 'E':
			cell = 'D' + cell
		cellsinfo[cell] = {'x2': [None, None, None], 'x0': [None, None, None]}
	print(cellsinfo.keys())
	for session in sessions:
		cellname = session.cell
		gain = session.gain
		slope = session.slope
		pvalue = session.pvalue
		tlength = session.tlength.keys()[0]
		if tlength == '250 ms':
			cellsinfo[cellname][gain][0] = slope
		elif tlength == '500 ms':
			cellsinfo[cellname][gain][1] = slope
		else:
			cellsinfo[cellname][gain][2] = slope
	return cellsinfo

def add_one_plot_traces(fig, sessions, sheetinfo):
	cellsinfo = createcellsinfo(sessions, sheetinfo)
	y_up = [[], [], []]
	y_down = [[],[],[]]
	x_up = sorted(cellsinfo.keys())
	x_down =sorted(cellsinfo.keys())
	gains = ['x2', 'x0']
	neg_sig_pvalues_up = [[], [], []]
	neg_sig_pvalues_down = [[], [], []]
	tlengths = ['250 ms', '500 ms', '1000 ms']
	for cell in x_up:
		for t, tlength in enumerate(tlengths):
			slope = cellsinfo[cell]['x2'][t]
			pvalue = cellsinfo[cell]['x2'][t]
			y_up[t].append(slope)

	for cell in x_down:
		for t, tlength in enumerate(tlengths):
			slope = cellsinfo[cell]['x0'][t]
			pvalue = cellsinfo[cell]['x2'][t]
			y_down[t].append(slope)
	fig.add_trace(go.Scatter(
		x = x_up,
		y = y_up[0],
		marker = dict(color = '#1b9e77', size = 8.5, symbol = 'circle'),
		mode = 'markers',
		name = '250 ms')
	)
	fig.add_trace(go.Scatter(
		x = x_up,
		y = y_up[1],
		marker = dict(color = '#d95f02', size = 8.5, symbol = 'square'),
		mode = 'markers',
		name = '500 ms')
	)
	fig.add_trace(go.Scatter(
		x = x_up,
		y = y_up[2],
		marker = dict(color = '#7570b3', size = 8.5, symbol = 'triangle-up'),
		mode = 'markers',
		name = '1000 ms')
	)
	fig.add_trace(go.Scatter(
		x = x_down,
		y = y_down[0],
		marker = dict(color = '#1b9e77', size = 8.5, symbol = 'circle'),
		mode = 'markers',
		name = '250 ms')
	)
	fig.add_trace(go.Scatter(
		x = x_down,
		y = y_down[1],
		marker = dict(color = '#d95f02', size = 8.5, symbol = 'square'),
		mode = 'markers',
		name = '500 ms')
	)
	fig.add_trace(go.Scatter(
		x = x_down,
		y = y_down[2],
		marker = dict(color = '#7570b3', size = 8.5, symbol = 'triangle-up'),
		mode = 'markers',
		name = '1000 ms')
	)
	trace_names = ["Up", "Up", "Up", "Down", "Down", "Down"]
	return trace_names



def make_scatter_line_trace(session, trace_name, numcells, sheetinfo, plot_params):
	session_df = session.data
	trials = session_df['Trial Starts']
	fr_baseline_subtracted = session_df['Firing Rate'] - session_df['Baseline Firing Rate']
	best_fit = session_df['Best Fit Line']
	scatter_trace = None
	line_trace = None
	if len(plot_params) == 0:
		scatter_trace = go.Scatter(
			name = "Firing Rate", \
			x = trials, 
			y = fr_baseline_subtracted, 
			mode = 'markers'
		)	
		line_trace = go.Scatter(
			x = trials,
	 		y = best_fit, 
	 		mode = 'lines', 
	 		name = "Slope = " + str(round(session.slope, 4)) + "\n" + str(session.pvalue)
	 	)
		
	else:
		color, symbol, label = get_plot_attributes(trace_name, plot_params)
		scatter_trace = go.Scatter(
			name = "Firing Rate " + label, 
			x = trials, 
			y = fr_baseline_subtracted, 
			mode = 'lines+markers', 
			opacity = 0.5, 
			marker = dict(symbol = symbol, color = color, opacity = 0.75), 
			line = dict(dash = 'dot', color = color),
			showlegend = True,
			hoverinfo = 'x+y'
		)
		
		line_trace = go.Scatter(
			x = trials,
 			y = best_fit, 
 			mode = 'lines', 
 			line = dict(color = color), 
 			name =  "m" + label  + (" = " + str(round(session.slope, 3)) + ", p = " + str(round(session.pvalue, 3))),
 			hoverinfo = 'name',
 			hoverlabel = dict(namelength = -1),

 		)
 	return scatter_trace, line_trace



def make_average_trace(fig, average_plot, plot_params):
	for (row, col) in average_plot.keys():
		cell_names = []
		cell_slopes = []
		pos_sig_pvalues = []
		neg_sig_pvalues = []
		for t, trace_value in enumerate(average_plot[(row, col)]):
			#if trace_value[0] == 'E32-2' and trace_value[1] == '1000 ms' and trace_value[2] == 'x2' : continue
			textval = ''
			if "allconditions" in plot_params:
				textval = trace_value[0]
			elif "comparegains" in plot_params:
				textval = trace_value[0] + ", " + trace_value[1]
			elif "comparelengths" in plot_params:
				textval = trace_value[0] + ", " + trace_value[2]
			textval = textval + ", p-value: " + str(round(trace_value[4], 4))
			if trace_value[4] < 0.05:
				neg_sig_pvalues.append(t)

			cell_slopes.append(trace_value[3])
			cell_names.append(textval)
		box = go.Box(
			name = 'Summary Plot',
			y = cell_slopes,
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

	

def add_traces(fig,sessions, sheetinfo, numcells, plot_params):
	if "oneplotsummaries" in plot_params: 
		return add_one_plot_traces(fig, sessions, sheetinfo)
	trace_names = []
	average_plot = {}
	for session in sessions:
		trace_name = (session.cell, session.tlength.keys()[0], session.gain)
		scatter_trace, line_trace = make_scatter_line_trace(session, trace_name, numcells, sheetinfo, plot_params)
		if len(plot_params) == 0:
			fig.add_trace(scatter_trace)
			fig.add_trace(line_trace)	
		else:
			plot_row = 0
			plot_col = 0
			if "allconditions" in plot_params:
				if session.gain == 'x2':
					plot_row = 1
				else:
					plot_row = 2
				if session.tlength.keys()[0] == '250 ms':
					plot_col = 1
				elif session.tlength.keys()[0] == '500 ms':
					plot_col = 2
				elif session.tlength.keys()[0] == '1000 ms':
					plot_col = 3
			else:
				plot_row = 1
				if "comparelengths" in plot_params:
					plot_col = session.tlength[session.tlength.keys()[0]] + 1
				elif "comparegains" in plot_params:
					plot_col = 1 if session.gain == 'x2' else 2
			fig.append_trace(scatter_trace, plot_row, plot_col)
			fig.append_trace(line_trace, plot_row, plot_col)
			trace_value = trace_name + (session.slope,session.pvalue)
			print(trace_value)
			if (plot_row, plot_col) in average_plot:
				average_plot[(plot_row, plot_col)].append(trace_value)
			else:
				average_plot[(plot_row, plot_col)] = [trace_value]

		trace_names = trace_names + [trace_name, trace_name]
	make_average_trace(fig, average_plot, plot_params)

	return trace_names


#Fix This Mess
def get_annotations(fig, name, sheetinfo, plot_params, average_plot_names = None):
	annotations = []
	plot_names = []
	if average_plot_names != None:
		for n, name in enumerate(average_plot_names):
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
	elif "allconditions" in plot_params:
		annotations = []
		plot_names = []
		gain_names = [' Gain Up (x2) Ipsi', ' Gain Down (x0) Contra']
		if "oppside" in plot_params:
			gain_names = [' Gain Up (x2) Contra', ' Gain Down (x0) Ipsi']
		length_names = [" 250 ms", " 500 ms", " 1000 ms"]
		plot_names = [name + length_names[j] + gain_names[i] for i in range(len(gain_names)) for j in range(len(length_names))]
		for i in range(len(plot_names)):
			title_xcoord = (fig['layout']['xaxis' + str(i+1)]['domain'][0] + fig['layout']['xaxis' + str(i+1)]['domain'][1])/2
			title_ycoord = fig['layout']['yaxis' + str(i+1)]['domain'][1]
			note = dict(
					font = dict(size = 14), 
					xref = "paper", 
					yref = "paper", 
					xanchor = "center", 
					yanchor = "bottom",
					x = title_xcoord,
					y = title_ycoord,
					text = plot_names[i], 
					showarrow = False)
			annotations.append(note)
	else:
		plot_names = []
		if "comparelengths" in plot_params:
			plot_names = ["250 ms Trials", "500 ms Trials", "1000 ms Trials"]
		elif "comparegains" in plot_params:
			if "oppside" not in plot_params:
				plot_names = ['Gain Up (x2) Ipsiversive Trials', 'Gain Down (x0) Contraversive Trials']	
			else:
				plot_names = ['Gain Up (x2) Contraversive Trials', 'Gain Down (x0) Ipsiversive Trials']
		for i in range(len(plot_names)):
			title_xcoord = (fig['layout']['xaxis' + str(i+1)]['domain'][0] + fig['layout']['xaxis' + str(i+1)]['domain'][1])/2
			title_ycoord = fig['layout']['yaxis' + str(i+1)]['domain'][1]
			note = dict(
					font = dict(size = 14), 
					xref = "paper", 
					yref = "paper", 
					xanchor = "center", 
					yanchor = "bottom",
					x = title_xcoord,
					y = title_ycoord,
					text = plot_names[i], 
					showarrow = False)
			annotations.append(note)
	return annotations

def make_average_button(annotations, trace_names, average_plot_names):
	print(average_plot_names)
	visibility = [False for _ in range(len(trace_names))] + [True for _ in range(len(average_plot_names))] #Fix Magic Number
	average_button = dict(
					label = "Summary Box Plot",
					method = 'update',
					args = [
						dict(
						visible = visibility
						), 
						dict(
						title = dict(
								text = "<b> Summary Box Plot </b> <br> Baseline Subtracted PC Firing Rate vs. Time",
								font = dict(
										size = 16
										),	
								y = 0.98,
								x = 0.495,
								xanchor = 'center',
								yanchor = 'top'
								),
						annotations = annotations
						)
						]
					)
	return average_button

def make_oneplot_buttons(fig, dropdown_names, trace_names):
	buttons = []
	print(trace_names)
	for n, name in enumerate(dropdown_names):
		visibility = []
		if n == 0:
			visibility = [True if trace == "Up" else False for trace in trace_names]
		else:
			visibility = [False if trace == "Up" else True for trace in trace_names]
		button =  dict(
					label = name,
					method = 'update',
					args = [
						dict(
						visible = visibility
						), 
						dict(
						title = dict(
								text = "<b> Summary Cell Plot </b> <br> Baseline Subtracted PC Firing Rate vs. Time",
								font = dict(
										size = 16
										),	
								y = 0.98,
								x = 0.495,
								xanchor = 'center',
								yanchor = 'top'
								)
						)
						]
					)
		buttons.append(button)
	return buttons



def make_buttons(fig, dropdown_names, trace_names, plot_params, sheetinfo, average_plot_names):
	if "oneplotsummaries" in plot_params:
		return make_oneplot_buttons(fig, dropdown_names, trace_names)
	buttons = []
	for n, name in enumerate(dropdown_names):
		print(name)
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

		annotations = get_annotations(fig, name, sheetinfo, plot_params)
		button = dict(
				label = label,
				method = 'update', 
				args = [{'visible': visibility}, 
						 {'title': {
							'text': "<b>" + label + "</b>" + "<br> Baseline Subtracted PC Firing Rate vs. Time",
							'font': {'size' : 16},
							'y': 0.98,
							'x': 0.495, 
							'xanchor': 'center',
							'yanchor': 'top'
							},
						  'annotations' : annotations
						}
						]
			)
		buttons.append(button)
	average_annotations = get_annotations(fig, "average", sheetinfo, plot_params, average_plot_names)
	average_button = make_average_button(average_annotations, trace_names, average_plot_names)
	buttons.insert(0,average_button)
	return buttons

def get_names(sessions, plot_params):
	dropdown_names = []
	average_plot_names = []
	filename = ""
	if len(plot_params) == 0:
		dropdown_names = [[session.cell, session.tlength.keys()[0], session.gain] for session in sessions]
		filename = "default"
	elif "oneplotsummaries" in plot_params:
		dropdown_names.append("Cells Gain Up" + (" (Ipsi)" if "oppside" not in plot_params else " (Contra)"))
		dropdown_names.append("Cells Gain Down" + (" (Contra)" if "oppside" not in plot_params else " (Ipsi)"))
		filename = "one-plot-summaries"
	else:
		gain_names = [' Gain Up (x2) Ipsi',  ' Gain Down (x0) Contra']
		length_names = [" 250 ms", " 500 ms", " 1000 ms"]
		if "allconditions" in plot_params:
			if "oppside" in plot_params:
				gain_names = [' Gain Up (x2) Contra', ' Gain Down (x0) Ipsi']
				length_names = [" 250 ms", " 500 ms", " 1000 ms"]
				filename = "oppside-"
			average_plot_names = [length_names[j] + gain_names[i] for i in range(len(gain_names)) for j in range(len(length_names))]
			dropdown_names = sorted(list(set([session.cell for session in sessions])))
			filename = filename + "allconditions-by-cell"
	 	elif "comparelengths" in plot_params and "comparegains" in plot_params:
 			dropdown_names = sorted(unique([session.cell for session in sessions]))
 			filename = "compare-lengths-and-gains-by-cell"
	 	elif "comparelengths" in plot_params:
			dropdown_names = list(set([(session.cell, session.gain) for session in sessions]))
			filename = "compare-lengths-by-cell"
			average_plot_names = length_names
			dropdown_names.sort(key = lambda x: (x[0], -int(x[1][1])))

		elif "comparegains" in plot_params:
			average_plot_names = gain_names
			dropdown_names = list(set([(session.cell, session.tlength.keys()[0], session.tlength[session.tlength.keys()[0]]) for session in sessions]))	
			filename = "compare-gains-by-cell"
			dropdown_names.sort(key = lambda x: (x[0], x[2]))
			dropdown_names = [(name[0], name[1]) for name in dropdown_names]

	return dropdown_names, average_plot_names, filename

def set_fig_dimensions(filename, numcells_each, plot_params):
	fig = go.Figure()
	numrows = 1
	numcols = 1
	plot_height = 650
	if filename == "default" or filename == "one-plot-summaries":
		return fig, plot_height	
	if "by-cell" in filename:
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
	elif "by-monkey" in filename:
		if "compare-lengths" in filename:
		 	numrows = 3
		 	plot_height = 850
		elif "compare-gains" in filename:
		 	numrows = 2
		 	plot_height = 750
		numcols = numcells_each	
	fig = subplots.make_subplots(rows = numrows, 
								 cols = numcols, 
								 horizontal_spacing = 0.025
								 )
	if numrows != 1:
		fig.update_xaxes(row = numrows, title_text = "Time (Seconds)")
	else: 
		fig.update_xaxes(title_text = "Time (Seconds)")
	fig.update_xaxes(tickmode = 'auto', range = [0, 90])

	if numcols != 1:
		fig.update_yaxes(col = 1, title_text = "PC Firing Rate - BLS (sp/s)")
	else:
		fig.update_yaxes(title_text = "PC Firing Rate - BLS (sp/s)")
	fig.update_yaxes(tickmode = 'auto', range = [-100, 280])

	return fig, plot_height




def plot_firing_rates(sessions, sheetinfo, plot_params):
	print("Number of Data Points: ", len(sessions))
	sessions.sort(key = lambda x: (x.cell, -int(x.gain[1]), x.tlength[x.tlength.keys()[0]]))
	dropdown_names, average_plot_names, filename = get_names(sessions, plot_params)
	numcells_each = len(sheetinfo.uniquecells)/2
	fig, plot_height = set_fig_dimensions(filename, numcells_each, plot_params)
	trace_names = add_traces(fig, sessions, sheetinfo, numcells_each, plot_params)
	buttons = make_buttons(fig, dropdown_names, trace_names, plot_params, sheetinfo, average_plot_names)
	button_layer_1_height = 1.10
	fig.update_layout(
		autosize = False,
		height = plot_height,
		width = 1605,
		margin = go.layout.Margin(l = 10, r = 10, b = 10, t = 15, pad = 3),
		yaxis_title = "PC Firing Rate - Baseline Subtracted (sp/s)",
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












