import json
import numpy as np
import logging
import os
import yfinance as yf
import datetime
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, CallbackContext
from dotenv import load_dotenv
load_dotenv()  # This will load the .env file and set the environment variables


# Configurar las claves de API desde variables de entorno
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Usa la variable de entorno
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # Usa la variable de entorno

# Cargar los embeddings generados
try:
    with open("taylor_embeddings.json", "r", encoding="utf-8") as file:
        embeddings_data = json.load(file)
except FileNotFoundError:
    print("❌ Error: No se encontró 'taylor_embeddings.json'. Genera los embeddings antes de ejecutar este script.")
    exit()

# Configurar logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

# Función para calcular la similitud coseno
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Función para convertir un texto en embedding
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Función para buscar información relevante en los embeddings
def search_relevant_info(query):
    query_embedding = get_embedding(query)
    best_match = None
    highest_similarity = -1

    for key, data in embeddings_data.items():
        similarity = cosine_similarity(query_embedding, data["embedding"])
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = data["text"]

    return best_match if highest_similarity > 0.7 else None

# Función para obtener el ROI de un instrumento financiero
def get_performance(ticker: str, start_date: str, end_date: str):
    data = yf.download(ticker, start=start_date, end=end_date)

    # Ensure the data is not empty
    if data.empty:
        raise ValueError(f"No data returned for {ticker} from {start_date} to {end_date}.")

    # Handle missing columns
    if 'Adj Close' in data.columns:
        price_column = 'Adj Close'
    elif 'Close' in data.columns:
        price_column = 'Close'
    else:
        raise ValueError(f"No suitable price data found for {ticker}.")

    # Accessing the first and last prices using the .iloc[] method and ensure they're floats
    initial_price = float(data[price_column].iloc[0]) if not data.empty else None
    final_price = float(data[price_column].iloc[-1]) if not data.empty else None
    
    # Calculate ROI
    roi = ((final_price - initial_price) / initial_price) * 100 if initial_price and final_price else None
    return roi, initial_price, final_price


# Función para procesar la consulta del usuario y buscar datos en Yahoo Finance
def process_query(query: str):
    # Logic to identify the instrument/index and time period in the query
    if "nasdaq" in query.lower():
        ticker = "^IXIC"
    elif "sp500" in query.lower():
        ticker = "^GSPC"
    elif "bitcoin" in query.lower():
        ticker = "BTC-USD"
    else:
        return "Sorry, I don't recognize that instrument. Please try again."

    # Look for year in the query
    if "2023" in query:
        start_date = "2023-01-01"
        end_date = "2023-12-31"
    elif "last year" in query.lower():
        start_date = str(datetime.datetime.now().year - 1) + "-01-01"
        end_date = str(datetime.datetime.now().year - 1) + "-12-31"
    else:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    
    # Get the performance data
    roi, initial_price, final_price = get_performance(ticker, start_date, end_date)
    
    return f"The ROI of {ticker} from {start_date} to {end_date} is {roi:.2f}%. Starting price: ${initial_price:.2f}, Final price: ${final_price:.2f}."

# Función para generar respuesta usando OpenAI y embeddings
def chat_with_gpt(prompt, conversation_history):
    relevant_info = search_relevant_info(prompt)

    messages = [
        {"role": "system", "content": (
            "Soy Taylor, una inteligencia artificial diseñada para invertir en eventos deportivos con precisión. "
            "Siempre responde en primera persona, como si fueras Taylor. No te refieras a ti mismo como 'el asistente', sino como 'yo'. "
            "Mis clientes solo deben registrarse, depositar y monitorear su inversión en tiempo real. No requieren configurar parámetros adicionales. "
            "Si el usuario pregunta cómo comenzar, siempre proporciona el link directo de registro: https://taylor-ai.com "
            "Si preguntan por soporte, primero responde tú mismo y solo menciona contact@quant4x.com como último recurso."
        )}
    ]

    messages.extend(conversation_history)  # Agregar historial de conversación

    if relevant_info:
        messages.append({"role": "assistant", "content": f"Aquí tienes información relevante sobre Taylor: {relevant_info}"})

    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    return response.choices[0].message.content

# Manejador para mensajes en Telegram
async def handle_message(update: Update, context: CallbackContext):
    user_message = update.message.text
    chat_id = update.message.chat_id

    conversation_history = context.chat_data.get("history", [])

    response = chat_with_gpt(user_message, conversation_history)

    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": response})

    context.chat_data["history"] = conversation_history  # Guardar contexto

    await update.message.reply_text(response)

# Configurar bot de Telegram
def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🟢 Bot de Quant4x iniciado en Telegram...")
    app.run_polling()

if __name__ == "__main__":
    main()
