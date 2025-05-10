# agents/tools/weather_tool.py
from .base_tool import Tool, ToolException
import requests
import json

class WeatherTool(Tool):
    name = "weather_lookup"
    description = ("Fetches current weather information for a specified location. "
                   "This tool provides temperature, conditions, and basic forecast data "
                   "to inform decisions or responses that depend on weather context.")
    argument_schema = {
        "location": "str: The city name or location to get weather for (e.g., 'London', 'New York, US')",
        "units": "str (optional): The temperature unit system to use - 'metric' (Celsius) or 'imperial' (Fahrenheit). Defaults to 'metric'."
    }

    def __init__(self, api_key=None):
        super().__init__()
        self.api_key = api_key or "4d5dcebd5295b73e4ab1b4a7ef30d9ba"  # Example API key placeholder
        self.api_base_url = "https://api.openweathermap.org/data/2.5/weather"

    def execute(self, **kwargs) -> str:
        location = kwargs.get("location")
        units = kwargs.get("units", "metric")
        
        if not location:
            raise ToolException("WeatherTool: 'location' parameter is required.")
        
        if units not in ["metric", "imperial"]:
            raise ToolException("WeatherTool: 'units' must be either 'metric' or 'imperial'.")
        
        try:
            print(f"[WeatherTool] Fetching weather for location: '{location}', units: '{units}'")
            
            # Prepare the request
            params = {
                "q": location,
                "appid": self.api_key,
                "units": units
            }
            
            # Make the API call
            response = requests.get(self.api_base_url, params=params, timeout=10)
            
            # Check for successful response
            if response.status_code == 200:
                data = response.json()
                
                # Extract and format the relevant information
                temp = data.get("main", {}).get("temp")
                feels_like = data.get("main", {}).get("feels_like")
                temp_min = data.get("main", {}).get("temp_min")
                temp_max = data.get("main", {}).get("temp_max")
                humidity = data.get("main", {}).get("humidity")
                
                weather_desc = data.get("weather", [{}])[0].get("description", "No description available")
                wind_speed = data.get("wind", {}).get("speed")
                
                # Build the response
                temp_unit = "°C" if units == "metric" else "°F"
                wind_unit = "m/s" if units == "metric" else "mph"
                
                weather_info = [
                    f"Weather in {data.get('name', location)}, {data.get('sys', {}).get('country', '')}:",
                    f"• Conditions: {weather_desc.capitalize()}",
                    f"• Temperature: {temp}{temp_unit} (feels like {feels_like}{temp_unit})",
                    f"• Range: {temp_min}{temp_unit} to {temp_max}{temp_unit}",
                    f"• Humidity: {humidity}%",
                    f"• Wind: {wind_speed} {wind_unit}"
                ]
                
                return "\n".join(weather_info)
            
            elif response.status_code == 401:
                raise ToolException("WeatherTool: API key is invalid or unauthorized. Please check your API key configuration.")
            
            elif response.status_code == 404:
                return f"Weather information for '{location}' could not be found. Please check if the location name is correct."
            
            else:
                raise ToolException(f"WeatherTool: API request failed with status code {response.status_code}: {response.text}")
                
        except requests.RequestException as e:
            print(f"[WeatherTool] Network error: {e}")
            raise ToolException(f"WeatherTool: Network error occurred while fetching weather data: {str(e)}")
        
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[WeatherTool] Error parsing response: {e}")
            raise ToolException(f"WeatherTool: Error parsing weather data response: {str(e)}")
        
        except Exception as e:
            print(f"[WeatherTool] Unexpected error: {e}")
            raise ToolException(f"WeatherTool: Unexpected error: {str(e)}")