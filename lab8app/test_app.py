import requests


data = {"data": [[0, 25847, 2, 6, 2, 1, 0.0, 1.0]]}
url = 'http://127.0.0.1:8000/predict'
response = requests.post(url, json=data)
print(response.json())