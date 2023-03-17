from pathlib import Path

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = Path(BASE_DIR,"data")
MODELS_DIR = Path(BASE_DIR,"models")

base_url = 'https://api.themoviedb.org/3/'
time_url = 'http://api.timezonedb.com/v2.1/get-time-zone?key=9TP1AB145H0I&format=xml&by=zone&zone=Asia/Manila'