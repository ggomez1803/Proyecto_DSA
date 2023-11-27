import json
from typing import Any
import typing as t
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib

from pydantic import BaseModel, ValidationError


from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger
#from model import __version__ as model_version
#from model.predict import make_prediction
#from model.processing.data_manager import load_pipeline
#from model.processing.validation import validate_inputs



from core import config

TRAINED_MODEL_DIR = "../model/"

import schemas
model_version = "0.0.1"
api_router = APIRouter()



class DataInputSchema(BaseModel):
    Churn: Optional[float]
    Efectividad_cobro: Optional[float]
    DLTV: Optional[float]
    

# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
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
                        "Efectividad_cobro": 0.98,
                        "DLTV": 2000000,
                    }
                ]
            }
        }


# Ruta para verificar que la API se esté ejecutando correctamente
@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name="Proyecto Segmentacion", api_version="0.0.1", model_version=model_version
    )

    return health.dict()

# Ruta para realizar las predicciones
@api_router.post("/predict", response_model=PredictionResults, status_code=200)
async def predict(input_data: MultipleDataInputs) -> Any:
    """
    Prediccion usando el modelo de bankchurn
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    logger.info(f"Making prediction on inputs: {input_data.inputs}")
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Prediction results: {results.get('predictions')}")

    return results

def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = "../model/modelo_segmentacion.pkl"
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
_abandono_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": model_version, "errors": errors}

    if not errors:
        predictions = _abandono_pipe.predict(
            X=validated_data[config.model_config.features]
        )
        results = {
            "predictions": [pred for pred in predictions], 
            "version": model_version,
            "errors": errors,
        }

    return results