import json
import numpy as np
import logging
import os
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, CallbackContext

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

# Función para generar respuesta usando OpenAI y embeddings

def chat_with_gpt(prompt, conversation_history):
    relevant_info = search_relevant_info(prompt)
    
    messages = [
        {"role": "system", "content": (
            "Eres Taylor, un asistente de Quant4x especializado en responder sobre Taylor, "
            "un servicio de inteligencia artificial para inversiones en apuestas deportivas. "
            "Siempre responde en el contexto de Taylor y evita respuestas genéricas sobre inversión. "
            "Si el usuario pregunta sobre cómo empezar, depósitos o retiros, proporciona detalles específicos sobre Taylor. "
            "Si el usuario pregunta cuánto puede ganar, usa datos reales y calcula la proyección de manera clara."
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
