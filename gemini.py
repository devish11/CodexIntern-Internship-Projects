import os
import requests
from env import GEMINI_API_KEY , GOOGLE_SEARCH_API_KEY , SEARCH_ENGINE_ID
import google.generativeai as genai


   # Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")  # Use the latest model

   # Keywords to detect real-time queries (expand as needed)
REAL_TIME_KEYWORDS = ["current", "price", "weather", "today", "now", "live"]

def is_real_time_query(user_input):
       """
       Checks if the user input likely requires real-time data based on keywords.
       
       Args:
           user_input (str): The user's message.
       
       Returns:
           bool: True if real-time data is needed, False otherwise.
       """
       return any(keyword in user_input.lower() for keyword in REAL_TIME_KEYWORDS)

def fetch_search_results(query):
       """
       Fetches search results from Google Custom Search API.
       
       Args:
           query (str): The search query.
       
       Returns:
           str: A summary of the top search results, or an error message.
       
       Raises:
           Exception: If the API request fails.
       """
       api_key = GOOGLE_SEARCH_API_KEY
       search_engine_id = SEARCH_ENGINE_ID
       url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={query}&num=3"
       
       try:
           response = requests.get(url)
           response.raise_for_status()
           data = response.json()
           results = data.get("items", [])
           if not results:
               return "No relevant search results found."
           
           # Summarize top 3 results
           summary = "\n".join([f"- {item['title']}: {item['snippet']}" for item in results[:3]])
           return f"Real-time data from search:\n{summary}"
       except requests.RequestException as e:
           raise Exception(f"Error fetching search results: {str(e)}")

def generate_response(user_input, chat_session):
       """
       Generates a response using Gemini, incorporating real-time data if needed.
       
       Args:
           user_input (str): The user's message.
           chat_session: The Gemini ChatSession object for history.
       
       Returns:
           str: The assistant's response.
       
       Raises:
           Exception: If Gemini API fails.
       """
       prompt = user_input
       
       if is_real_time_query(user_input):
           try:
               search_data = fetch_search_results(user_input)
               prompt = f"{user_input}\n\n{search_data}"
           except Exception as e:
               prompt = f"{user_input}\n\nNote: Could not fetch real-time data due to error: {str(e)}"
       
       try:
           response = chat_session.send_message(prompt)
           return response.text
       except Exception as e:
           raise Exception(f"Error generating response with Gemini: {str(e)}")

def main():
       """
       Main function to run the chatbot interactively.
       """
       print("Intelligent Chatbot with Gemini and Real-Time Search")
       print("Type 'exit' to quit.")
       
       # Start a new chat session
       chat_session = model.start_chat(history=[])
       
       while True:
           user_input = input("You: ").strip()
           if user_input.lower() == "exit":
               print("Goodbye!")
               break
           
           try:
               response = generate_response(user_input, chat_session)
               print(f"Assistant: {response}")
           except Exception as e:
               print(f"Error: {str(e)}. Please try again.")

if __name__ == "__main__":
       main()
