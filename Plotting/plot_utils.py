import plotly.graph_objects as go
import pandas as pd
from Structures.Objects import SessionInfo

GAINS = ['x2', 'x0']
GAIN_NAMES = ['Gain Up (x2) Ipsi', 'Gain Down (x0) Contra']
GAIN_NAMES_OPP = ['Gain Up (x2) Contra', 'Gain Down (x0) Ipsi']
COLORS = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']
LENGTHS = ['250 ms', '500 ms', '1000 ms']
SYMBOLS = ['circle', 'square', 'triangle-up']


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
		slope = session.slopes['FR']
		pvalue = session.pvalues['FR']
		tlength = session.tlength[0]
		if tlength == '250 ms':
			cellsinfo[cellname][gain][0] = slope
		elif tlength == '500 ms':
			cellsinfo[cellname][gain][1] = slope
		else:
			cellsinfo[cellname][gain][2] = slope
	return cellsinfo

def get_plot_attributes(plot_params, tlength, gain):
	plot_row = 0
	plot_col = 0
	color = ''
	symbol = ''
	label = ''
	legend_group = ''
	if "allconditions" in plot_params:
		if tlength == '250 ms':
			plot_col = 1
			if gain == 'x2':
				plot_row = 1
				color = COLORS[0]
			elif gain == 'x0':
				plot_row = 2
				color = COLORS[1]
			symbol = 'circle'
		elif tlength == '500 ms':
			plot_col = 2
			if gain == 'x2':
				plot_row = 1
				color = COLORS[2]
			elif gain == 'x0':
				plot_row = 2
				color = COLORS[3]
			symbol = 'triangle-up'
		elif tlength == '1000 ms':
			plot_col = 3
			if gain == 'x2':
				plot_row = 1
				color = COLORS[4]
			elif gain == 'x0':
				plot_row = 2
				color = COLORS[5]
			symbol = 'cross'
	if "comparelengths" in plot_params:
		plot_row = 1
		plot_col = 1
		color = COLORS[1]
		symbol = 'circle'
		legend_group = 'group1'
		if tlength == '500 ms':
			plot_col = 2
			color = COLORS[2]
			symbol = 'triangle-up'
			legend_group = 'group2'
		elif tlength == '1000 ms':
			plot_col = 3
			color = COLORS[3]
			symbol = 'cross'
			legend_group = 'group3'
		label = tlength
	elif "comparegains" in plot_params:
		plot_row = 1
		plot_col = 1
		color = COLORS[1]
		symbol = 'circle'
		legend_group = 'group1'
		label = gain + (" (ipsi)" if "oppside" not in plot_params else " (contra)")
		if gain == 'x0':
			plot_col = 2
			color = COLORS[2]
			symbol = 'cross'
			legend_group = 'group2'
			label = gain + (" (contra)" if "oppside" not in plot_params else " (ipsi)")
	plot_attributes = {"row": plot_row, "col": plot_col, "color" : color, "symbol" : symbol, "label" : label}
	return plot_attributes

def create_scatter_trace(fig, session, plot_params, plot_attributes):
	session_df = session.data
	trials = session_df['Trial Starts']
	fr_baseline_subtracted = session_df['Firing Rate'] - session_df['Baseline Firing Rate']
	scatter_trace = None
	
	color = plot_attributes['color']
	label = plot_attributes['label']
	symbol = plot_attributes['symbol']
	plot_row = plot_attributes['row']
	plot_col = plot_attributes['col']
	if len(plot_params) == 0 or (len(plot_params) == 1 and "allcells" in plot_params):
		scatter_trace = pltutils.create_scatter_trace()
		fig.append_trace(go.Scatter(
			name = "Firing Rate", \
			x = trials, 
			y = fr_baseline_subtracted, 
			mode = 'markers'
		))	
	else:
		scatter_trace = go.Scatter(
			name = "Firing Rate " + label, 
			x = trials, 
			y = fr_baseline_subtracted, 
			mode = 'lines+markers', 
			opacity = 0.5, 
			marker = dict(symbol = symbol, opacity = 0.75), 
			line = dict(dash = 'dot', color = color),
			showlegend = True,
			hoverinfo = 'x+y'
		)
		fig.append_trace(scatter_trace, plot_row, plot_col)

def create_button(label, visibility, annotations):
	button = 	dict(
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
	return button


def create_line_trace(fig, session, plot_params, plot_attributes):
	session_df = session.data
	trials = session_df['Trial Starts']
	best_fit = session_df['FR Best Fit Line']	
	line_trace = None

	color = plot_attributes['color']
	label = plot_attributes['label']
	symbol = plot_attributes['symbol']
	plot_row = plot_attributes['row']
	plot_col = plot_attributes['col']

	if len(plot_params) == 0 or (len(plot_params) == 1 and "allcells" in plot_params):
		fig.append(go.Scatter(
			x = trials,
 			y = best_fit, 
 			mode = 'lines', 
 			line = dict(color =  color), 
 			name =  "m " +  label 
 					+ (" = " + str(round(session.slopes['FR'], 3)) + ", p = " + str(round(session.pvalues['FR'], 3))),
 			hoverinfo = 'name',
 			hoverlabel = dict(namelength = -1),
 		))
 	else: 
 		line_trace = go.Scatter(
			x = trials,
 			y = best_fit, 
 			mode = 'lines', 
 			line = dict(color =  color), 
 			name =  "m " +  label 
 					+ (" = " + str(round(session.slopes['FR'], 3)) + ", p = " + str(round(session.pvalues['FR'], 3))),
 			hoverinfo = 'name',
 			hoverlabel = dict(namelength = -1),
 		)
 		plot_row = plot_attributes['row']
 		plot_col = plot_attributes['col']
 		fig.append_trace(line_trace, plot_row, plot_col)







