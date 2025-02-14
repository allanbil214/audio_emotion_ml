import requests

url = "http://127.0.0.1:8000/predict/"  # Change to your API endpoint
file_path = "C:/Users/Allan/Desktop/audify/datatest/OAF_back_ps.wav"  # Replace with your actual audio file

# Open and send the file
with open(file_path, "rb") as file:
    files = {"file": (file_path, file, "audio/wav")}
    response = requests.post(url, files=files)

# Print response
print(response.json())
