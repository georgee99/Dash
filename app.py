# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import html, dcc, dash_table
import pandas as pd
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import dash_bootstrap_components as dbc

df = pd.read_csv('table.csv')

def create_table(dataframe):
    return dash_table.DataTable(
            id='table-output',
            columns=[{'id': c, 'name': c} for c in dataframe.columns[:-1]],
            filter_action='native',
            editable=True,
            page_size=18,
            style_header={
                'fontWeight': 'bold',
            },
            style_cell={'textAlign': 'left'},
            style_table={'width': '90%', 'margin': '0 auto'}
    )

def line_graph():
    return dcc.Graph(
        id='line_graph', style={'height': '70vh'}
    )

def bar_graph():
    return dcc.Graph(
        id='bar_graph', style={'height': '70vh'}
    )

def method_dropdown():
    return html.Div([
        html.Label(['Select a Forecast methodology method'], style={'font-weight': 'bold', "text-align": "center", "margin-right": "10px"}),
        html.Abbr("?", title="These methods are Machine Learning models used to forecast "
                             "electricity prices and demand."),
        dcc.Dropdown(id='dropdown-methodology',
                     options=[{'label': m, 'value': m} for m in
                              df.sort_values('METHOD')['METHOD'].unique()],
                     value='ARIMA',
                     multi=False,
                     disabled=False,
                     clearable=False,
                     searchable=True,
                     placeholder='Choose Methodology...'),
    ])

def state_dropdown():
    return html.Div([
        html.Label(['Select a state'], style={'font-weight': 'bold', "text-align": "center"}),
        dcc.Dropdown(id='dropdown',
                     options=[{'label': s, 'value': s} for s in
                              df.sort_values('REGION')['REGION'].unique()],
                     value='NSW',
                     multi=False,
                     disabled=False,
                     clearable=False,
                     searchable=True,
                     placeholder='Choose State...'),

    ], style={'margin-right': '40px'})

explanation_text = 'This web application forecasts the price and demand of electricity in Australia. You can filter out ' \
                   'the data by "State" and "Machine Learning Methodology" using the dropdowns.'
explanation_text2 = 'You can also toggle between the visualisations by using the tabs. The visualisations include a Line ' \
                    'Graph, Bar Chart, and a Table View. The visualisations contain two y-axes. The Price axis (' \
                    '$AUD) is listed on the left, and Demand Axis (MW) is listed on the right.'
explanation_text3 = 'In the table visualisation, you can also search for certain data including date, price, and demand.'
explanation_text4 = 'The time interval has been set to 3 seconds. The forecast line is visible at the date: 30/05/2022.'

def pop_up_modal():
    return html.Div(
        [
            dbc.Button("How to use", id="modal_button", n_clicks=0, className='button_modal'),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("How to use this application")),
                    dbc.ModalBody(html.Div([
                        explanation_text,
                        html.Br(),
                        html.Br(),
                        explanation_text2,
                        html.Br(),
                        html.Br(),
                        explanation_text3,
                        html.Br(),
                        html.Br(),
                        explanation_text4
                    ])),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Close", id="close_button", className="ms-auto", n_clicks=0
                        )
                    ),
                ],
                id="explanation_modal",
                size='md',
                is_open=False,
                backdrop=True,
                fade=True,
        )
    ], style={'margin': '0 auto'})

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "UTS Capstone Project 12900825"

# Styling the tabs

tab_styling = {
    'border': '1px solid #bab5b5',
    'backgroundColor': 'white',
}

tab_selected_styling = {
    'borderTop': '1px solid #bab5b5',
    'backgroundColor': '#1fa3ff',
    'color': 'white',
}

tab_parent_styling = {
    'margin': '0 auto',
    'width': '90%',
    'font-family': 'cursive',
}

app.layout = html.Div([
    html.Div([
        html.H1('Electricity Price and Demand Forecast', className="app-header"),
        pop_up_modal(),
    ], style={'display': 'grid', 'margin-bottom': '10px'}),

    html.Div([
        state_dropdown(),
        method_dropdown(),
    ], style={'display': 'grid', 'grid-template-columns': 'repeat(2, 0.25fr)', 'margin-left': '4rem', 'margin-bottom': '20px'}),

    html.Br(),

    dcc.Tabs([
        dcc.Tab(label='Line Graph View', style=tab_styling, selected_style=tab_selected_styling, children=[
            line_graph()
        ]),
        dcc.Tab(label='Bar Chart View', style=tab_styling, selected_style=tab_selected_styling, children=[
            bar_graph()
        ]),
        dcc.Tab(label='Table View', style=tab_styling, selected_style=tab_selected_styling, children=[
            html.Br(),
            create_table(df)
        ], )
    ], style=tab_parent_styling),

    dcc.Interval(
        id='interval-component',
        interval=3 * 1000,  # in milliseconds
        n_intervals=10,
        max_intervals=1920  # DataFrame index for 30/06/2022
    ),
    html.H3('George El-Zakhem 12900825', style={'margin': '10px 2rem', 'font-size': '20px'}),
])

# Visualisation Callback
@app.callback(
    Output('line_graph','figure'),
    Output('bar_graph', 'figure'),
    Output('table-output', 'data'),
    [Input('interval-component', 'n_intervals')],
    [Input('dropdown','value')],
    [Input('dropdown-methodology', 'value')],
)

def update_figure(n_intervals, selected_state, selected_method):
    # Formatting the data
    filtered_df = df[(df.REGION == selected_state) & (df.METHOD == selected_method)].head(n_intervals)

    dates = filtered_df['SETTLEMENTDATE'].to_numpy()
    price = filtered_df['RRP'].to_numpy()
    demand = filtered_df['TOTALDEMAND'].to_numpy()

    min_price = np.amin(price)
    max_price = np.amax(price)

    min_demand = np.amin(demand)
    max_demand = np.amax(demand)

    # Line Graph below
    line_figure = make_subplots(specs=[[{"secondary_y": True}]])

    line_figure.add_trace(
        go.Scatter(x=dates[n_intervals-10 : n_intervals], y=price[n_intervals-10 : n_intervals], name="price", marker_color="blue",
            marker=dict(
                size=15,
                symbol='circle',
                line=dict(
                    color='Black',
                    width=2,
                )
        ),),
        secondary_y=False,
    )

    line_figure.add_trace(
        go.Scatter(x=dates[n_intervals-10 : n_intervals], y=demand[n_intervals-10 : n_intervals], name="demand", marker_color='purple',
            marker=dict(
                size=15,
                symbol='circle',
                line=dict(
                    color='Black',
                    width=2
                )
        ),),
        secondary_y=True,
    )

    line_figure.update_layout(
        title_text="Line Graph"
    )

    line_figure.update_xaxes(title_text="Time Period")
    line_figure.update_yaxes(title_text="<b>Price</b> ($AUD)", secondary_y=False, range=(min_price - 10, max_price + 10), constrain='domain')
    line_figure.update_yaxes(title_text="<b>Demand</b> (MW)", secondary_y=True, range=(min_demand - 15000, max_demand + 15000), constrain='domain')

    # Adding forecast line for line graph
    if '30/05/2022 0:00' in dates[-10:]:
        line_figure.add_vline(x='30/05/2022 0:00', line_dash="dash", line_width=3);
    else:
        pass

    # Bar Chart below
    bar_chart = go.Figure(
        data=[
            go.Bar(x=dates[n_intervals - 10: n_intervals], y=price[n_intervals - 10: n_intervals], name="price",
                   marker_color='blue', yaxis='y', offsetgroup=1, textposition='auto'),
            go.Bar(x=dates[n_intervals - 10: n_intervals], y=demand[n_intervals - 10: n_intervals], name="demand",
                   marker_color='purple', yaxis='y2', offsetgroup=2, textposition='auto')
        ],
        layout={
            'yaxis': {'title': '<b>Price</b> ($AUD)'},
            'yaxis2': {'title': '<b>Demand</b> (MW)', 'overlaying': 'y', 'side': 'right'},
            'xaxis': {'title': 'Time Period'}
        }
    )

    bar_chart.update_layout(barmode='group', title_text="Bar Chart")

    # Adding forecast line for bar chart
    if '30/05/2022 0:00' in dates[-10:]:
        bar_chart.add_vline(x='30/05/2022 0:00', line_dash="dash", line_width=14)
    else:
        pass

    # Table graph
    the_table = filtered_df.to_dict('records')

    return line_figure, bar_chart, the_table

# Modal Callback
@app.callback(
    Output("explanation_modal", "is_open"),
    [Input("modal_button", "n_clicks"), Input("close_button", "n_clicks")],
    [State("explanation_modal", "is_open")],
)

def update_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    else:
        return is_open

if __name__ == '__main__':
    app.run_server(debug=True)
