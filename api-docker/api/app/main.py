from typing import Any
import json
import typing as t
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib

from pydantic import BaseModel, ValidationError


from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger
from pydantic import AnyHttpUrl, BaseSettings

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger

#from api import api_router
#from config import settings, setup_app_logging

# setup logging as early as possible
#setup_app_logging(config=settings)


from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel
from strictyaml import YAML, load


# Project Directories
CONFIG_FILE_PATH = "/opt/api/app/config.yml"
API_V1_STR: str = "/api/v1"

class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    #data_file: str
    train_data_file: str
    test_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    features: List[str]
    test_size: float
    random_state: int
    k: int
    qual_vars: List[str]
    categorical_vars: Sequence[str]
    qual_mappings: Dict[str, int]


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    
    return CONFIG_FILE_PATH
    


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()

TRAINED_MODEL_DIR = "/opt/api/model/"

model_version = "0.0.1"
api_router = APIRouter()


app = FastAPI(
    title="Proyecto Segmentacion", openapi_url="/api/v1/openapi.json"
)

root_router = APIRouter()

# Cuerpo de la respuesta en la raíz
@root_router.get("/")
def index(request: Request) -> Any:
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)

#from model import __version__ as model_version
#from model.predict import make_prediction
#from model.processing.data_manager import load_pipeline
#from model.processing.validation import validate_inputs

class DataInputSchema(BaseModel):
    Churn: Optional[float]
    Efectividad_cobro: Optional[float]
    DLTV: Optional[float]
    


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]


# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
    errors: Optional[Any]
    predictions: Optional[List[float]]

# Esquema para inputs múltiples
class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "Churn": 0.01,
                        "DLTV": 2000000,
                        "Efectividad_cobro": 0.98,
                    }
                ]
            }
        }

class Health(BaseModel):
    name: str
    api_version: str
    model_version: str
# Ruta para verificar que la API se esté ejecutando correctamente
@api_router.get("/health", response_model=Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = Health(
        name="Proyecto Segmentacion", api_version="0.0.1", model_version=model_version
    )

    return health.dict()

# Ruta para realizar las predicciones
@api_router.post("/predict", response_model=PredictionResults, status_code=200)
async def predict(input_data: Dict) -> Any:
    print(input_data)
    """
    Prediccion usando el modelo de segmentación
    """
    #print(input_data)
    #input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

   

   # logger.info(f"Making prediction on inputs: {input_data.inputs}")
    results = make_prediction(input_data=input_data["inputs"])
    
    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Prediction results: {results.get('predictions')}")

    return results

def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = "/opt/api/model/modelo_segmentacion.pkl"
    trained_model = joblib.load(filename=file_path)
    return trained_model




def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_config.features
        if validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data

def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors






pipeline_file_name = "modelo_segmentacion.pkl"
modelo_segmentacion = load_pipeline(file_name=pipeline_file_name)


app.include_router(api_router, prefix=API_V1_STR)
app.include_router(root_router)
BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # type: ignore
        "http://localhost:8000",  # type: ignore
        "https://localhost:3000",  # type: ignore
        "https://localhost:8000",  # type: ignore
    ]
# Set all CORS enabled origins
if BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""
    print(input_data)
    data = pd.DataFrame(input_data)
   # validated_data, errors = validate_inputs(input_data=data)
   # print(validated_data)
   # print(errors)
    #results = {"predictions": None, "version": model_version, "errors": errors}

    #if not errors:
    #    predictions = modelo_segmentacion.predict(
    #        X=validated_data[config.model_config.features]
    #    )
     #   results = {
     #       "predictions": [pred for pred in predictions], 
      #      "version": model_version,
      #      "errors": errors,
      #  }
    #results = {"predictions": None, "errors": errors}
    #validated_data['DLTV_std'] = StandardScaler().fit_transform(validated_data[['DLTV']])
    #validated_data['VL_Churn_Prob_std'] = StandardScaler().fit_transform(validated_data[['Churn']])
    #validated_data['Efectividad_cobro_std'] = StandardScaler().fit_transform(validated_data[['Efectividad_cobro']])
    
    # Realizar predicción
    #cluster = modelo_segmentacion.predict(validated_data[['DLTV_std', 'VL_Churn_Prob_std', 'Efectividad_cobro_std']])
    dltv_mean = 2052875.1935207853
    dltv_std = 1106063.9693825627
    churn_mean = 0.7525753912711541
    churn_std = 0.5247159306383898
    efect_mean = 0.9211417379699978
    efect_std = 0.14987482287122253

    data['DLTV'] = (data['DLTV'] - dltv_mean) / dltv_std
    data['Churn'] = (data['Churn'] - churn_mean) / churn_std
    data['Efectividad_cobro'] = (data['Efectividad_cobro'] - efect_mean) / efect_std
    
    cluster = modelo_segmentacion.predict(data[['DLTV', 'Churn', 'Efectividad_cobro']])

    print(f'Se recibe el DLTV: {data["DLTV"]}')
    print(f'Se recibe el Churn: {data["Churn"]}')
    print(f'Se recibe el Efectividad_cobro: {data["Efectividad_cobro"]}')

    print(f'Se predice el cluster: {cluster}')

    results = {"predictions": [pred for pred in cluster], "errors": None}
    return results
