import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import datetime as dt

# Importar librerías necesarias para la predicción
import joblib
from sklearn.preprocessing import StandardScaler

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Dashboard Donantes Proyección Infantil"

server = app.server
app.config.suppress_callback_exceptions = True

# Cargar modelo de segmentación
segmentacion_modelo = joblib.load('kmeans_model.pkl')

# Load data from csv
def load_data():
    # To do: Completar la función 
    data = pd.read_csv('./Archivos_Cliente/Base_lfv.csv', encoding='latin1')
    return data

# Función para escalar datos
def scale_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Cargar datos
data = load_data()

app.layout = html.Div(
    id="app-container",
    children=[
        # Descripción y título
        html.Div(
            id="description-card",
            children=[
                html.H3("Pronóstico de segmentación de donantes"),
                html.Div(
                    id="intro",
                    children="Esta herramienta contiene información sobre la segmentación de donantes de una organización sin fines de lucro. "
                ),
            ],
        ),

        # Controles
        html.Div(
            id="control-card",
            children=[
                # Nuevos campos de entrada para DLTV y Fuga
                html.P("Ingresar valores para predicción:"),
                html.Div(
                    id="componentes-prediccion",
                    children=[
                        html.Div(
                            id="componente-dltv",
                            children=[
                                dcc.Input(
                                    id="input-dltv",
                                    type="number",
                                    value=0,
                                    placeholder="DLTV",
                                ),
                            ],
                            style=dict(width='30%')
                        ),

                        html.P(" ", style=dict(width='5%', textAlign='center')),

                        html.Div(
                            id="componente-fuga",
                            children=[
                                dcc.Input(
                                    id="input-fuga",
                                    type="number",
                                    value=0,
                                    placeholder="Fuga",
                                ),
                            ],
                            style=dict(width='30%')
                        ),

                        html.P(" ", style=dict(width='5%', textAlign='center')),

                        html.Div(
                            id="componente-boton",
                            children=[
                                html.Button(
                                    id="button-prediccion",
                                    n_clicks=0,
                                    children="Realizar Predicción",
                                    style={"fontSize": 14},
                                ),
                            ],
                            style=dict(width='20%')
                        ),
                    ],
                    style=dict(display='flex')
                ),
            ],
        ),

        # Gráfica de la serie de tiempo
        html.Div(
            id="model_graph",
            children=[
                html.B("Gráfica de donantes por segmento según su valor y probabilidad de fuga"),
                html.Hr(),
                dcc.Graph(
                    id="plot_series",
                )
            ],
        ),
    ],
)


def plot_series(data, initial_date, proy):
    data_plot = data.loc[initial_date:]
    data_plot = data_plot[:-(120 - proy)]
    fig = go.Figure([
        go.Scatter(
            name='Demanda energética',
            x=data_plot.index,
            y=data_plot['AT_load_actual_entsoe_transparency'],
            mode='lines',
            line=dict(color="#188463"),
        ),
        go.Scatter(
            name='Proyección',
            x=data_plot.index,
            y=data_plot['forecast'],
            mode='lines',
            line=dict(color="#bbffeb",),
        ),
        go.Scatter(
            name='Upper Bound',
            x=data_plot.index,
            y=data_plot['Upper bound'],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=data_plot.index,
            y=data_plot['Lower bound'],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor="rgba(242, 255, 251, 0.3)",
            fill='tonexty',
            showlegend=False
        )
    ])

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis_title='Demanda total [MW]',
        hovermode="x"
    )
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#2cfec1")
    fig.update_xaxes(showgrid=True, gridwidth=0.25, gridcolor='#7C7C7C')
    fig.update_yaxes(showgrid=True, gridwidth=0.25, gridcolor='#7C7C7C')

    return fig

# Predicción de cluster
def predict_cluster(dltv, fuga):
    # Escalar datos
    scaled_input = scale_data(np.array([[dltv, fuga]]))
    # Realizar predicción
    cluster_predicho = segmentacion_modelo.predict(scaled_input)
    return cluster_predicho[0]

def generate_control_card():
    return html.Div(
        id="control-card",
        children=[
            # ... (tu código existente)

            # Nuevos campos de entrada para DLTV y Fuga
            html.P("Ingresar valores para predicción:"),
            html.Div(   
                id="componentes-prediccion",
                children=[
                    html.Div(
                        id="componente-dltv",
                        children=[
                            dcc.Input(
                                id="input-dltv",
                                type="number",
                                value=0,
                                placeholder="DLTV",
                            ),
                        ],
                        style=dict(width='30%')
                    ),
                    
                    html.P("qué pasa si escribo aquí???",style=dict(width='5%', textAlign='center')),
                    
                    html.Div(
                        id="componente-fuga",
                        children=[
                            dcc.Input(
                                id="input-fuga",
                                type="number",
                                value=0,
                                placeholder="Fuga",
                            ),
                        ],
                        style=dict(width='30%')
                    ),
                    
                    html.P("aquí hay otro campo vacío",style=dict(width='5%', textAlign='center')),
                    
                    html.Div(
                        id="componente-boton",
                        children=[
                            html.Button(
                                id="button-prediccion",
                                n_clicks=0,
                                children="Realizar Predicción",
                                style={"fontSize": 14},
                            ),
                        ],
                        style=dict(width='20%')
                    ),
                ],
                style=dict(display='flex')
            ),

            # ... (tu código existente)
        ]
    )

# ... (tu código existente)

# Callback para realizar la predicción y actualizar la gráfica
@app.callback(
    [Output(component_id="plot_series", component_property="figure")],
    [Input(component_id="button-prediccion", component_property="n_clicks")],
    [State(component_id="input-dltv", component_property="value"),
     State(component_id="input-fuga", component_property="value")]
)
def update_output_div(date, hour, proy, n_clicks, dltv, fuga):
    if ((date is not None) & (hour is not None) & (proy is not None)):
        hour = str(hour)
        minute = str(0)

        initial_date = date + " " + hour + ":" + minute
        initial_date = pd.to_datetime(initial_date, format="%Y-%m-%d %H:%M")

        # Generar la figura directamente llamando a la función plot_series
        fig = plot_series(data, initial_date, int(proy))

        # Realizar la predicción solo si se hizo clic en el botón de predicción
        if n_clicks > 0:
            # Realizar predicción de cluster
            cluster_predicho = predict_cluster(dltv, fuga)
            mensaje_prediccion = f"La observación pertenece al cluster {cluster_predicho}"

            return fig, mensaje_prediccion
        else:
            return fig, ""


# Run the server
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8080, debug=True)
