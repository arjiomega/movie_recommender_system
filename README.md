

# RECOMMENDER SYSTEM

This is a movie recommender system that uses your current list of watched movies to recommend other titles you may be interested in. It is currently using content based recommender system.

## Virtual Environment Setup

```bash
conda create --no-default-packages -n <env_name>
conda activate <env_name>
conda install python=3.9
```

## Install Packages

```bash
pip install -r requirements.txt
```

## Create config.py in main directory
Follow the instructions here to generate your own api key:
<br>
https://developers.themoviedb.org/3/getting-started/introduction
```python
tmdb_api_key = '<api_key>'
```


## Run fastAPI using uvicorn
```bash
uvicorn app.api:app --host localhost --port 8000 --reload --reload-dir .
```

## Run Movie Recommendation using python
The list of movies for a user needs to be the tmdb_id for a specific movie. It is made this way to prevent any problems caused by the case of a string (Ex. The Dark Knight != the dark knight).
```python
import requests
import json
payload = {
            "user_id": 0,
            "movie_list": [76600, 267805, 315162, 436270, 505642, 536554, 587092, 631842, 640146, 646389, 653851, 758009, 785084, 823999, 842544, 842942, 843794, 1058949]
}
response = requests.post(url='http://localhost:8000/predict',json=payload)
output_ = json.loads(response.text)
```