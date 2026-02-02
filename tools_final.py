"""
Tools module for LangGraph chatbot with real API integrations
- Weather: OpenWeather API (with geocoding)
- QA/Search: Tavily API
- Stocks: Alpha Vantage API
"""

import os
import requests
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# WEATHER TOOL - Using OpenWeather API
# ============================================================================

def get_weather(location: str) -> str:
    """
    Fetch real weather data using OpenWeather API
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    if not api_key:
        return "❌ OpenWeather API key not configured."
    
    try:
        # Geocoding - convert city to coordinates
        geocoding_url = "https://api.openweathermap.org/geo/1.0/direct"
        geo_params = {
            "q": location,
            "appid": api_key,
            "limit": 1
        }
        
        print(f"🔍 Geocoding '{location}'...")
        geo_response = requests.get(geocoding_url, params=geo_params, timeout=5)
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        
        if not geo_data:
            return f"❌ City '{location}' not found."
        
        result = geo_data[0]
        latitude = result["lat"]
        longitude = result["lon"]
        city = result["name"]
        country = result.get("country", "")
        
        # Fetch weather data
        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        weather_params = {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric"
        }
        
        print(f"🌍 Fetching weather for {city}...")
        weather_response = requests.get(weather_url, params=weather_params, timeout=5)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        
        # Extract weather information
        current = weather_data.get("main", {})
        weather_desc = weather_data.get("weather", [{}])[0].get("description", "Unknown")
        wind = weather_data.get("wind", {})
        clouds = weather_data.get("clouds", {})
        
        temperature = current.get("temp", "N/A")
        feels_like = current.get("feels_like", "N/A")
        humidity = current.get("humidity", "N/A")
        pressure = current.get("pressure", "N/A")
        wind_speed = wind.get("speed", "N/A")
        wind_deg = wind.get("deg", "N/A")
        cloudiness = clouds.get("all", "N/A")
        visibility = weather_data.get("visibility", "N/A")
        
        # Get wind direction
        wind_directions = {
            "N": (337.5, 22.5),
            "NE": (22.5, 67.5),
            "E": (67.5, 112.5),
            "SE": (112.5, 157.5),
            "S": (157.5, 202.5),
            "SW": (202.5, 247.5),
            "W": (247.5, 292.5),
            "NW": (292.5, 337.5)
        }
        
        wind_dir = "N/A"
        if isinstance(wind_deg, (int, float)):
            for direction, (min_deg, max_deg) in wind_directions.items():
                if min_deg <= wind_deg < max_deg or (direction == "N" and (wind_deg >= 337.5 or wind_deg < 22.5)):
                    wind_dir = direction
                    break
        
        result_str = f"""
🌍 **Weather in {city}, {country}**

📊 Current Conditions:
   • Temperature: {temperature}°C (Feels like: {feels_like}°C)
   • Condition: {weather_desc.capitalize()}
   • Humidity: {humidity}%
   • Pressure: {pressure} hPa
   • Wind Speed: {wind_speed} m/s ({wind_dir})
   • Cloud Coverage: {cloudiness}%
   • Visibility: {visibility} m
"""
        return result_str.strip()
        
    except Exception as e:
        return f"❌ Error fetching weather: {str(e)}"


# ============================================================================
# STOCK TOOL - Using Alpha Vantage API (FIXED)
# ============================================================================

def get_stock_price(ticker: str) -> str:
    """
    Fetch real stock price data using Alpha Vantage API
    FIXED: Better error handling and response parsing
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    
    if not api_key:
        return "❌ Stock API key not configured."
    
    try:
        ticker_upper = ticker.upper().strip()
        
        print(f"💰 Fetching stock price for '{ticker_upper}'...")
        print(f"   API Key: {api_key[:10]}...")  # Show first 10 chars
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker_upper,
            "apikey": api_key
        }
        
        print(f"   URL: {url}")
        print(f"   Params: {params}")
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print(f"   Response keys: {list(data.keys())}")
        print(f"   Full response: {data}")
        
        # Check for API errors
        if "Error Message" in data:
            error_msg = data["Error Message"]
            print(f"❌ API Error: {error_msg}")
            return f"❌ Invalid ticker: {ticker}\nError: {error_msg}"
        
        if "Note" in data:
            note_msg = data["Note"]
            print(f"⚠️ API Note: {note_msg}")
            return f"❌ API rate limited. {note_msg}\n\nWait 60 seconds and try again."
        
        # Get Global Quote
        quote = data.get("Global Quote", {})
        
        print(f"   Global Quote: {quote}")
        
        # Check if we got data
        if not quote or len(quote) == 0:
            print(f"❌ Empty quote data")
            return f"❌ No data for ticker: {ticker}\n\nTry again or wait 60 seconds (rate limit)."
        
        # Extract fields
        symbol = quote.get("01. symbol")
        price = quote.get("05. price")
        change = quote.get("09. change")
        change_percent = quote.get("10. change percent")
        high = quote.get("03. high")
        low = quote.get("04. low")
        volume = quote.get("06. volume")
        timestamp = quote.get("07. latest trading day")
        
        print(f"✅ Got stock data: Symbol={symbol}, Price={price}")
        
        # Validate we got a price
        if not price or price == "0" or price == "None":
            print(f"❌ Invalid price: {price}")
            return f"❌ No price available for {ticker}.\n\nWait 60 seconds (API rate limit) and try again."
        
        # Format price as float for safety
        try:
            price_float = float(price)
        except:
            return f"❌ Invalid price format: {price}"
        
        # Determine trend
        try:
            change_float = float(change.replace("+", "") if change else "0")
            trend = "📈" if change_float > 0 else "📉" if change_float < 0 else "➡️"
        except:
            trend = "📊"
        
        result_str = f"""
💰 **Stock Quote - {symbol}**

📊 Price Information:
   • Current Price: ${price}
   • Change: {change} ({change_percent}) {trend}
   • 52-Week High: ${high}
   • 52-Week Low: ${low}
   • Volume: {volume}
   • Last Update: {timestamp}

⏰ Data from Alpha Vantage
"""
        print(f"✅ Returning formatted stock data")
        return result_str.strip()
        
    except requests.exceptions.Timeout:
        print(f"❌ Request timeout")
        return "❌ Request timed out. Try again."
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection error")
        return "❌ Connection error. Check internet."
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return f"❌ Error: {str(e)}"


# ============================================================================
# GENERAL QA TOOL - Using Tavily API
# ============================================================================

def get_answer_tavily(query: str) -> str:
    """
    Get answers using Tavily API for search and QA
    """
    api_key = os.getenv("TAVILY_API_KEY")
    
    if not api_key:
        return "❌ Tavily API key not configured."
    
    try:
        url = "https://api.tavily.com/search"
        
        payload = {
            "api_key": api_key,
            "query": query,
            "include_answer": True,
            "max_results": 3,
            "search_depth": "basic"
        }
        
        print(f"🔍 Searching for '{query}'...")
        
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract answer and results
        answer = data.get("answer", "No answer found")
        results = data.get("results", [])
        
        # Format response
        result_str = f"""
📚 **Answer:**

{answer}

📰 **Sources:**
"""
        
        if results:
            for idx, result in enumerate(results[:3], 1):
                title = result.get("title", "Untitled")
                source = result.get("url", "")
                content = result.get("content", "")
                
                result_str += f"\n{idx}. {title}"
                if content:
                    result_str += f"\n   {content[:150]}..."
                if source:
                    result_str += f"\n   {source}"
        
        return result_str.strip()
        
    except Exception as e:
        return f"❌ Error searching: {str(e)}"


# ============================================================================
# LANGCHAIN RUNNABLE WRAPPERS
# ============================================================================

from langchain_core.runnables import RunnableLambda

# Create runnable versions for LangGraph
tool1_weather = RunnableLambda(lambda location: get_weather(location))
tool2_stock = RunnableLambda(lambda ticker: get_stock_price(ticker))
tool3_qa = RunnableLambda(lambda query: get_answer_tavily(query))