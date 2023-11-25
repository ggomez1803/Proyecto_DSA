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
def scale_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Predicción de cluster
def predict_cluster(fuga, efect, dltv):
    # Escalar datos
    scaled_input = scale_data(np.array([[fuga, efect, dltv]]))
    # Realizar predicción
    cluster_predicho = modelo_seg.predict(scaled_input)
    return cluster_predicho[0]

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
                html.H6("Ingrese la información del donante:"),
                html.Div(["Probabilidad de fuga del donante: ",
                          dcc.Input(id='fuga', value='0', type='number')]),
                html.Br(),
                html.Div(["Donor lifetime value del donante: ",
                          dcc.Input(id='dltv', value='35000', type='number')]),
                html.Br(),
                html.Div(["Efectividad de cobro: ",
                          dcc.Input(id='efect', value='0', type='number')]),
                html.Br(),
                html.H6(html.Div(id='resultado')),
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

        # Add prediction section
        html.Div(
            id="prediction-card",
            children=[
                html.H2("Predicción del segmento"),
                html.P("El donante pertenece al cluster:"),
                html.Div(id="predicted-cluster"),
            ],
        ),
    ],
)

# Method to update prediction
@app.callback(
    [Output(component_id='resultado', component_property='children'),
    Output(component_id='plot_series', component_property='figure'),
    Output(component_id='predicted-cluster', component_property='children')],
    [Input(component_id='fuga', component_property='value'), 
     Input(component_id='dltv', component_property='value'), 
     Input(component_id='efect', component_property='value')]
)
def update_output_div(fuga, efect, dltv):
    figure = None
    document = None
    myreq = {
        "inputs": [
            {
            "Churn": int(fuga),
            "Efectividad_cobro": int(efect),
            "DLTV": int(dltv)
            }
        ]
      }
    headers =  {"Content-Type":"application/json", "accept": "application/json"}

    # POST call to the API
    response = requests.post(api_url, data=json.dumps(myreq), headers=headers)
    data = response.json()
    logger.info("Response: {}".format(data))

    # Pick result to return from json format
    cluster = predict_cluster(float(fuga), float(efect), float(dltv))
    
    if data["predictions"][0] == 0:
        cluster = "Campeones"
    elif data["predictions"][0] == 1:
        cluster = "Comprometidos"
    elif data["predictions"][0] == 2:
        cluster = "Hibernando"
    elif data["predictions"][0] == 3:
        cluster = "En riesgo"
    elif data["predictions"][0] == 4:
        cluster = "En fuga"
    else:
        cluster = "No se puede clasificar"

    result = f'El donante pertenece al cluster: {cluster}'
    logger.info("Result: {}".format(result))

    predicted_cluster_display = f"Pertenece al cluster: {cluster}"
    # Update figure
    figure = dcc.Graph(
    id="plot_series",
    figure=figure,
    style={"height": "100%", "width": "100%"},
    config={"displayModeBar": False})

    document.getElementById('predicted-cluster').innerHTML = predicted_cluster_display
    return result, figure, predicted_cluster_display
 

# Run the server
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8001, debug=True)
