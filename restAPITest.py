import requests
import webbrowser
import os

def control_shelly_light(state="ON", auth_token="oh.ToggleLight.i994VVCmkKJgzwUanlkAFP1Pi86QpajtliS9OYdETG1vBh1c58DQTnZa0mjXp95MS8KBpYn7Fu5GtYRgbiaQ"):
    url = "http://10.100.91.14:8080/rest/items/ShellyLight_Betrieb"
    
    headers = {
        "Content-Type": "text/plain"
    }
    
    # Add auth token if provided
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    try:
        # Send POST request with the state as data
        response = requests.post(url, data=state, headers=headers)
        
        if response.status_code == 200:
            print(f"✓ Light successfully set to {state}")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
        else:
            print(f"✗ Failed to control light")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
        return response
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error occurred: {e}")
        return None

def test_api(url):
    """
    Sends a GET request to the specified URL and prints the response.
    """
    try:
        # Send a GET request to the API endpoint
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            print("API is working!")
            print("Status Code:", response.status_code)
            
            # Check if the content is HTML and open the URL in the browser
            if 'text/html' in response.headers.get('Content-Type', ''):
                print("Response is HTML. Opening URL in browser.")
                with open("response.html", "w", encoding="utf-8") as f:
                    f.write(response.text)
                print("Response saved to response.html")
                # To open the local file instead of the URL, you could use:
                # webbrowser.open('file://' + os.path.realpath("response.html"))
                webbrowser.open(url)
            else:
                # Try to print the response content as JSON
                try:
                    print("Response JSON:", response.json())
                except requests.exceptions.JSONDecodeError:
                    print("Response content is not in JSON format.")
                    print("Response Text:", response.text)
        else:
            print(f"API call failed with status code: {response.status_code}")
            print("Response Text:", response.text)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Control the Shelly light
    print("=== Turning Light ON ===")
    control_shelly_light(state="ON")


    
    # You can also turn it off
    # print("\n=== Turning Light OFF ===")
    # control_shelly_light(state="OFF", auth_token="your_token_here")
    
    # Original test code (commented out)
    # ip_address = '10.100.91.14'  
    # port = '8080'            
    # endpoint = '/'    # Accessing the base URL
    # api_url_local = f"http://{ip_address}:{port}{endpoint}"
    # print(f"Testing API at: {api_url_local}")
    # test_api(api_url_local)

