import json
import os
import numpy as np
from openai import OpenAI

# Cargar la clave de OpenAI desde variables de entorno
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Usa la variable de entorno

# Cargar las FAQs desde el archivo JSON (ahora usando el nombre correcto)
try:
    with open("taylor_faqs.json", "r", encoding="utf-8") as file:
        faqs = json.load(file)
except FileNotFoundError:
    print("❌ Error: No se encontró 'taylor_faqs.json'. Asegúrate de que el archivo existe antes de generar embeddings.")
    exit()

# Función para obtener los embeddings
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding
