# app/api.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from http import HTTPStatus
from typing import Dict
from functools import wraps
from datetime import datetime

import pickle
from pathlib import Path
import sys
sys.path.append(Path(__file__).parent.parent.absolute())
from main import RecommendationSystem

# for aws lambda
from mangum import Mangum

# Define application
app = FastAPI(
    title="Movie Recommendation",
    description="Recommend movie using recommender system",
    version="0.1",
)

handler = Mangum(app)

def construct_response(f):
    """Construct a JSON response for an endpoint."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> Dict:
        # results store the return value of the original function 'f'
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return JSONResponse(response)

    return wrap


## LOADS EVERYTHING
# @app.on_event("startup")
# def load_artifacts():
#     global artifacts
#     run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
#     artifacts = main.load_artifacts(model_dir=config.MODEL_DIR)
#     logger.info("Ready for inference!")


@app.get("/", tags=["General"])
@construct_response
# _index gets overwritten by the decorator
def _index(request: Request) -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response

from pathlib import Path


@app.get("/predict", tags=["predict"])
@construct_response
def predict_(request: Request, movie:str):
    rs = RecommendationSystem()
    rs.add_movie(movie)
    output_ = rs.recommend()

    #data = {"predict": output_.to_dict(orient='records')}
    data = {"predict": output_.to_dict()}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": "Test2",
    }
    return response

# @app.get("/performance", tags=["Performance"])
# @construct_response
# def _performance(request: Request, filter: str = None) -> Dict:
#     """Get the performance metrics."""
#     performance = artifacts["performance"]
#     data = {"performance":performance.get(filter, performance)}
#     response = {
#         "message": HTTPStatus.OK.phrase,
#         "status-code": HTTPStatus.OK,
#         "data": data,
#     }
#     return response

