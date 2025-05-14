import requests
import time

while True:
    try:
        response = requests.get("https://ipapi.co/json/", timeout=5)
        data = response.json()

        latitude = float(data['latitude'])
        longitude = float(data['longitude'])

        print(f"Latitude: {latitude:.7f}, Longitude: {longitude:.7f}")

    except Exception as e:
        print(f"Error fetching location: {e}")

    time.sleep(10)