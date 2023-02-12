import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={'N': 2, 'P': 9, 'K': 6, 'temperature': 20.87974371, 'humidity': 82.00274423, 'ph': 6.502985292, 'rainfall': 202.9355362})

print(r.json())