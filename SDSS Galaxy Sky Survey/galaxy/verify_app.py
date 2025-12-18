
import requests
import sys
import time

def verify():
    url = "http://127.0.0.1:2222/predict"
    data = {
        'u': 19.4,
        'g': 18.2,
        'r': 17.1,
        'i': 16.5,
        'z': 15.8,
        'redshift': 0.05
    }
    
    # Wait a bit for server to start
    time.sleep(2)
    
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            if "Galaxy Type:" in response.text:
                print("Verification SUCCESS: Prediction returned.")
                print("Response extract:", response.text[response.text.find("Galaxy Type:"):response.text.find("Galaxy Type:")+50])
            else:
                print("Verification FAILED: 'Galaxy Type:' not found in response.")
        else:
            print(f"Verification FAILED: Status code {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("Verification FAILED: Could not connect to server.")

if __name__ == "__main__":
    verify()
