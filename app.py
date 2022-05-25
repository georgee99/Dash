# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from dash.exceptions import PreventUpdate
import dash_table

# df = pd.read_csv('table.csv')
df = pd.read_csv('delete_this.csv')

# df = pd.read_csv('copy_of_final_table.csv')

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


def create_table(dataframe):
    return dash_table.DataTable(
            id='table-output',
            columns=[{'id': c, 'name': c} for c in dataframe.columns[:-1]],
            editable = True,
            # page_size = 30,
            # page_action='none',
            fixed_rows={'headers': True},
            style_table={'height': '1000'}
    )


def line_graph():
    return dcc.Graph(
        id='first_graph', style={'height': '70vh'}
    )

def bar_graph():
    return dcc.Graph(
        id='bar_graph', style={'height': '70vh'}
    )

def interval():
    return dcc.Interval(
        id='interval-component',
        interval=1*1000,
        n_intervals=0
    ),

def method_dropdown():
    return html.Div([
        html.Label(['Choose the Forecast method'], style={'font-weight': 'bold', "text-align": "center", "margin-right": "10px"}),
        html.Abbr("?", title="These methods are Machine Learning models used to forecast future "
                             "electricity prices and demand."),
        dcc.Dropdown(id='dropdown-methodology',
                     options=[{'label': x, 'value': x} for x in
                              df.sort_values('METHOD')['METHOD'].unique()],
                     value='ARIMA',
                     multi=False,
                     disabled=False,
                     clearable=False,
                     searchable=True,
                     placeholder='Choose Methodology...',
                     persistence='string',
                     persistence_type='memory'),
    ])

def state_dropdown():
    return html.Div([
        html.Label(['Choose the state you want'], style={'font-weight': 'bold', "text-align": "center"}),
        dcc.Dropdown(id='dropdown',
                     options=[{'label': x, 'value': x} for x in
                              df.sort_values('REGION')['REGION'].unique()],
                     value='NSW',
                     multi=False,
                     disabled=False,
                     clearable=False,
                     searchable=True,
                     placeholder='Choose State...',
                     persistence='string',
                     persistence_type='memory'),

    ], style={'margin-right': '40px'})

app = dash.Dash(__name__)
server = app.server

# Styling classes

tab_styling = {
    'borderBottom': '1px solid #d6d6d6',
}

tab_selected_styling = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
}

tab_parent_styling = {
    'margin': '0 auto',
    'width': '90%',
}


app.layout = html.Div([
    html.Div([
        html.H1('Electricity Price and Demand Forecast', style={'color': 'blue', 'margin': '20px auto 40px auto', 'font-family':"Verdana"}),
    ], style={'display': 'flex'}),
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
            # generate_table(df),
            html.Br(),
            create_table(df)
        ], )
    ], style=tab_parent_styling),

    dcc.Interval(
        id='interval-component',
        interval=3 * 1000,  # in milliseconds
        n_intervals=10
    ),
])

@app.callback(
    Output('first_graph','figure'),
    Output('bar_graph', 'figure'),
    Output('table-output', 'data'),
    [Input('interval-component', 'n_intervals')],
    [Input('dropdown','value')],
    [Input('dropdown-methodology', 'value')],
)

def update_figure(n_intervals, selected_state, selected_method):
    # Formatting the data
    filtered_df = df[(df.REGION == selected_state) & (df.METHOD == selected_method)].head(n_intervals)

    date_count = len(df[(df.REGION == selected_state) & (df.METHOD == selected_method)]['SETTLEMENTDATE'])

    dates = filtered_df['SETTLEMENTDATE'].to_numpy()
    price = filtered_df['RRP'].to_numpy()
    demand = filtered_df['TOTALDEMAND'].to_numpy()

    remainder = date_count - 10
    remainder_df = df[df.REGION == selected_state].tail(remainder)

    min_price = np.amin(price)
    max_price = np.amax(price)

    min_demand = np.amin(demand)
    max_demand = np.amax(demand)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=dates[n_intervals-10 : n_intervals], y=price[n_intervals-10 : n_intervals], name="price", marker_color="blue",
            marker=dict(
                size=15,
                line=dict(
                    color='Black',
                    width=2
                )
        ),),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=dates[n_intervals-10 : n_intervals], y=demand[n_intervals-10 : n_intervals], name="demand", marker_color='purple',
            marker=dict(
                size=15,
                line=dict(
                    color='Black',
                    width=2
                )
        ),),
        secondary_y=True,
    )

    fig.update_layout(
        title_text="Line Graph"
    )

    fig.update_xaxes(title_text="Time Period")
    fig.update_yaxes(title_text="<b>Price</b> $", secondary_y=False, range=(min_price - 10, max_price + 10), constrain='domain')
    fig.update_yaxes(title_text="<b>Demand</b> (MW)", secondary_y=True, range=(min_demand - 15000, max_demand + 15000), constrain='domain')

    # check if its in the last 10 elements of dates array
    if '1/01/2019 7:30' in dates[-10:]:
        fig.add_vline(x='1/01/2019 7:30', line_dash="dash", line_width=3);
    else:
        pass

    # Bar Chart below
    bar_chart = make_subplots(specs=[[{"secondary_y": True}]])

    bar_chart.add_trace(
        go.Bar(x=dates[n_intervals-10 : n_intervals], y=demand[n_intervals-10 : n_intervals], name="demand", marker_color='red',),
        secondary_y=False,
    )

    bar_chart.add_trace(
        go.Scatter(x=dates[n_intervals-10 : n_intervals], y=price[n_intervals-10 : n_intervals], name="price", marker_color='yellow',
            marker=dict(
                size=22.5,
                line=dict(
                    color='Black',
                    width=2,
                )
        ),),
        secondary_y=True,
    )

    # Add figure title
    bar_chart.update_layout(
        title_text="Bar Chart"
    )

    bar_chart.update_xaxes(title_text="Time Period")
    bar_chart.update_yaxes(title_text="<b>Price</b> $", secondary_y=False)
    bar_chart.update_yaxes(title_text="<b>Demand</b> units", secondary_y=True)

    # bar_chart.add_vline(x='1/01/2019 3:30', line_dash="dash", line_width=7);
    if '1/01/2019 7:30' in dates[-10:]:
        bar_chart.add_vline(x='1/01/2019 7:30', line_dash="dash", line_width=7);
    else:
        pass

    # Table graph

    the_table = filtered_df.to_dict('rows')

    if date_count - n_intervals < 1:
        raise PreventUpdate

    return fig, bar_chart, the_table

if __name__ == '__main__':
    app.run_server(debug=True)
