# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from dash.exceptions import PreventUpdate
import dash_table
from dash import no_update
import dash_bootstrap_components as dbc


# df = pd.read_csv('table.csv')
df = pd.read_csv('table_copy.csv')
# df = pd.read_csv('delete_this.csv')

# df = pd.read_csv('copy_of_final_table.csv')

def create_table(dataframe):
    return dash_table.DataTable(
            id='table-output',
            columns=[{'id': c, 'name': c} for c in dataframe.columns[:-2]],
            editable = True,
            page_size = 18,
            style_header={
                # 'backgroundColor': 'white',
                'fontWeight': 'bold',
            },
            style_cell={'textAlign': 'left'},
            style_table={'width': '90%', 'margin': '0 auto'}
    )


def line_graph():
    return dcc.Graph(
        id='first_graph', style={'height': '70vh'}
    )

def bar_graph():
    return dcc.Graph(
        id='bar_graph', style={'height': '70vh'}
    )

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
                     value='NSW1',
                     multi=False,
                     disabled=False,
                     clearable=False,
                     searchable=True,
                     placeholder='Choose State...',
                     persistence='string',
                     persistence_type='memory'),

    ], style={'margin-right': '40px'})

explanation_text = 'This web application forecasts the price and demand of electricity in Australia. You can filter out ' \
                   'the data by "State" and "Machine Learning Methodology" using the dropdowns. You can also toggle between ' \
                   'the visualisations by using the tabs. The visualisations include a Line Graph, Bar Chart, and a Table View. '

def pop_up_modal():
    return html.Div(
        [
            dbc.Button("How to use", id="modal_button", n_clicks=0, className='button_modal'),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("How to use this application")),
                    dbc.ModalBody(explanation_text),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Close", id="close_button", className="ms-auto", n_clicks=0
                        )
                    ),
                ],
                id="explanation_modal",
                is_open=False,
                size='md',
                backdrop=True,
                fade=True,
        )
    ], style={'margin': '0 auto'})

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "UTS Capstone Project 12900825"

# Styling classes

tab_styling = {
    'border': '1px solid #d6d6d6',
    'backgroundColor': 'white',

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
    'font-family': 'cursive',
}


app.layout = html.Div([
    html.Div([
        html.H1('Electricity Price and Demand Forecast', className="app-header"),
        pop_up_modal(),
    ], style={'display': 'grid'}),

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
        max_intervals=13 # set this to the length of the filtered dataframe
    ),

    html.H3('George El-Zakhem 12900825', style={'margin': '10px 2rem', 'font-size': '20px'}),
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
                symbol='circle',
                line=dict(
                    color='Black',
                    width=2,
                )
        ),),
        secondary_y=False,
    )

    fig.add_trace(
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
    bar_chart = go.Figure(
        data=[
            go.Bar(x=dates[n_intervals - 10: n_intervals], y=price[n_intervals - 10: n_intervals], name="price",
                   marker_color='blue', yaxis='y', offsetgroup=1, textposition='auto'),
            go.Bar(x=dates[n_intervals - 10: n_intervals], y=demand[n_intervals - 10: n_intervals], name="demand",
                   marker_color='purple', yaxis='y2', offsetgroup=2, textposition='auto')
        ],
        layout={
            'yaxis': {'title': '<b>Price</b> $'},
            'yaxis2': {'title': '<b>Demand</b> (MW)', 'overlaying': 'y', 'side': 'right'}
        }
    )

    bar_chart.update_layout(barmode='group', title_text="Bar Chart")

    if '1/01/2019 7:30' in dates[-10:]:
        bar_chart.add_vline(x='1/01/2019 7:30', line_dash="dash", line_width=7);
    else:
        pass

    # Table graph
    the_table = filtered_df.to_dict('rows')

    # if date_count - n_intervals < 0:
        # raise PreventUpdate
        # return no_update, '{} is prime!'.format(n_intervals)
    print(n_intervals)

    return fig, bar_chart, the_table

@app.callback(
    Output("explanation_modal", "is_open"),
    [Input("modal_button", "n_clicks"), Input("close_button", "n_clicks")],
    [State("explanation_modal", "is_open")],
)

def update_modall(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    else:
        return is_open

if __name__ == '__main__':
    app.run_server(debug=True)
