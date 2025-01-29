import json
import os
import numpy as np
from openai import OpenAI

# Cargar la clave de OpenAI desde variables de entorno
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Usa la variable de entorno

# Cargar las FAQs desde el archivo JSON
try:
    with open("faqs.json", "r", encoding="utf-8") as file:
        faqs = json.load(file)
except FileNotFoundError:
    print("❌ Error: No se encontró 'faqs.json'. Asegúrate de que el archivo existe antes de generar embeddings.")
    exit()

# Función para obtener los embeddings

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Generar embeddings para cada pregunta-respuesta en el archivo JSON
embeddings_data = {}

for key, value in faqs["taylor"].items():
    embeddings_data[key] = {
        "text": value,
        "embedding": get_embedding(value)  # Convertir la respuesta en embedding
    }

# Guardar los embeddings en un archivo JSON
with open("taylor_embeddings.json", "w", encoding="utf-8") as file:
    json.dump(embeddings_data, file, ensure_ascii=False, indent=4)

print("✅ Embeddings generados y guardados en 'taylor_embeddings.json'")