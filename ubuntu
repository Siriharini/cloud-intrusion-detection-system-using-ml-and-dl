import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "features": [0.1]*41   # dummy input (same size as dataset)
}

response = requests.post(url, json=data)

print(response.json())