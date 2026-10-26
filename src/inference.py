import os
import json
import time
import argparse
import ollama

def run_inference(model_name, suite_path, temperature, max_tokens):
    # Validar que exista el archivo de prompts
    if not os.path.exists(suite_path):
        print(f"❌ Error: El archivo {suite_path} no existe.")
        return

    with open(suite_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    os.makedirs('results', exist_ok=True)

    print(f"🚀 Iniciando inferencia con el modelo: {model_name}\n")

    for item in data['prompts']:
        print(f"Ejecutando prompt {item['id']} [{item['categoria']}]...")
        
        try:
            start_time = time.time()
            ttft = None
            full_response = ""
            
            # Usar streaming para calcular el Tiempo al Primer Token (TTFT)
            response_stream = ollama.generate(
                model=model_name,
                prompt=item['prompt'],
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens
                },
                stream=True
            )
            
            for chunk in response_stream:
                if ttft is None:
                    ttft = time.time() - start_time  # Mide el TTFT
                full_response += chunk['response']
                
            total_time = time.time() - start_time
            
            # Estimación simple de tokens por segundo (1 palabra ~ 1.3 tokens)
            estimated_tokens = int(len(full_response.split()) * 1.3)
            tokens_per_second = estimated_tokens / total_time if total_time > 0 else 0
            
            results.append({
                "id": item['id'],
                "categoria": item['categoria'],
                "prompt": item['prompt'],
                "salida": full_response,
                "ttft_segundos": round(ttft, 4) if ttft else 0,
                "tokens_por_segundo": round(tokens_per_second, 2),
                "tiempo_total_segundos": round(total_time, 2)
            })
            
        except Exception as e:
            print(f"❌ Error al conectar con Ollama: {e}")
            print("Asegúrate de que la aplicación Ollama esté abierta y corriendo de fondo.")
            return

    # Guardar resultados
    output_path = f"results/resultado_{model_name.replace(':', '_')}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ ¡Suite completada con éxito! Resultados guardados en: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Inferencia Local para Proyecto de IA")
    parser.add_argument('--model', type=str, default='mistral:7b-instruct-v0.3-q4_K_M', help='Nombre del modelo en Ollama')
    parser.add_argument('--suite', type=str, default='prompts/suite.json', help='Ruta al archivo JSON de prompts')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperatura de muestreo')
    parser.add_argument('--max_tokens', type=int, default=500, help='Máximo de tokens a generar')
    
    args = parser.parse_args()
    run_inference(args.model, args.suite, args.temperature, args.max_tokens)
