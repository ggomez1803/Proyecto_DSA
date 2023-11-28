import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output
import requests
import json
from loguru import logger
import os
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.graph_objs as go

# app server
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Dashboard Donantes Proyección Infantil"

server = app.server

app.config.suppress_callback_exceptions = True

# Cargar modelo de segmentación
modelo_seg = joblib.load('/opt/app/modelo_segmentacion.pkl')

# Función para escalar datos
#def scale_data(data):
#    scaler = StandardScaler()
#    scaled_data = scaler.fit_transform(data)
#    return scaled_data

# Predicción de cluster
#def predict_cluster(fuga, efect, dltv):
#    # Escalar datos
#    scaled_input = scale_data(np.array([[fuga, efect, dltv]]))
#    # Realizar predicción
#    cluster_predicho = modelo_seg.predict(scaled_input)
#    return cluster_predicho[0]

# PREDICTION API URL 
api_url = os.getenv('API_URL')
api_url = "http://{}:8001/api/v1/predict".format(api_url)

# Layout in HTML
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
                html.H3("Ingrese la información del donante:"),
                html.Div(["Probabilidad de fuga del donante (0-100): ",
                          dcc.Input(id='fuga', value='0', type='number')]),
                html.Br(),
                html.Div(["Donor lifetime value del donante:",
                          dcc.Input(id='dltv', value='0', type='number')]),
                html.Br(),
                html.Div(["Efectividad de cobro (0-1):",
                          dcc.Input(id='efect', value='0', type='number')]),
                html.Br(),
                # Botón para activar la predicción
                html.Button('Realizar Predicción', id='boton-prediccion', n_clicks=0),
                html.Br(),
                html.H2(html.Div(id='resultado')),
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

# Method to update prediction
@app.callback(
    [Output(component_id='resultado', component_property='children'),
    Output(component_id='plot_series', component_property='figure'),
    Output('boton-prediccion', 'n_clicks')],
    [Input(component_id='fuga', component_property='value'), 
     Input(component_id='dltv', component_property='value'), 
     Input(component_id='efect', component_property='value'),
     Input('boton-prediccion', 'n_clicks')]
)
def update_output_div(fuga, dltv, efect, n_clicks):
    # Inicializa el resultado y la figura
    result = None
    figure = None

    # Realiza la llamada al API solo cuando el botón ha sido clicado
    if n_clicks > 0:
        myreq = {
            "inputs": [
                {
                    "Churn": float(fuga),
                    "DLTV": float(dltv),
                    "Efectividad_cobro": float(efect)
                }
            ]
        }

        headers = {"Content-Type": "application/json", "accept": "application/json"}

        # POST call to the API
        response = requests.post(api_url, data=json.dumps(myreq), headers=headers)
        data = response.json()
        logger.info("Response: {}".format(data))
        print(logger.info("Response: {}".format(data)))

        cluster = ""

        if data["predictions"][0] == 0:
            cluster = "Hibernando"
        elif data["predictions"][0] == 1:
            cluster = "Comprometidos"
        elif data["predictions"][0] == 2:
            cluster = "En riesgo"
        elif data["predictions"][0] == 3:
            cluster = "Campeones"
        elif data["predictions"][0] == 4:
            cluster = "En fuga"
        else:
            cluster = "No se puede clasificar"

        result = f'El donante pertenece al cluster: {cluster}'
        logger.info("Result: {}".format(result))

    # Actualiza la figura
    #figure = dcc.Graph(
    #    id="plot_series",
    #    figure=figure,
    #    style={"height": "100%", "width": "100%"},
    #    config={"displayModeBar": False}
    #)

    # Construye el gráfico de dispersión
    scatter_trace = go.Scatter(
        x=[float(fuga)],
        y=[float(dltv)],
        mode='markers',
        marker=dict(size=10, color='blue'),  # puedes ajustar el tamaño y color
        name='Datos de entrada',
        text='Donante'  # etiqueta para los puntos
    )

    layout = go.Layout(
        title='Gráfico de Dispersión: Fuga vs DLTV',
        xaxis=dict(title='Probabilidad de Fuga'),
        yaxis=dict(title='Donor Lifetime Value')
    )

    figure = go.Figure(data=[scatter_trace], layout=layout)

    n_clicks = 0

    return result, figure, n_clicks
 

# Run the server
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8001, debug=True)
