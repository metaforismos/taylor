# test_yahoo.py

from bot_telegram import process_query  # Import the function from bot_telegram.py

# Test the function with a query
query = "What was the ROI of the NASDAQ in 2023?"
response = process_query(query)

# Print the response
print(response)
from dotenv import load_dotenv
load_dotenv()  # This will load the .env file and set the variables

import os
