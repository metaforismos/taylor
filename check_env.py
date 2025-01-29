from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

# Print to check if the token is being loaded correctly
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")

print("Telegram Token:", telegram_token)
print("OpenAI API Key:", openai_api_key)

if telegram_token is None or openai_api_key is None:
    print("❌ Error: One or more environment variables are not set properly.")
else:
    print("✅ Environment variables loaded successfully.")

