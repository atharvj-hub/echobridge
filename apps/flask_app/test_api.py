import requests

response = requests.post(
    'http://127.0.0.1:5000/api/emotion/analyze',
    json={'text': 'I am so happy today!'}
)

print(response.json())
