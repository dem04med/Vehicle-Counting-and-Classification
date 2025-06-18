from dotenv import load_dotenv
import openai
import os

# === CONFIGURAÇÕES ===

load_dotenv()  

openai.api_key = os.getenv("OPENROUTER_API_KEY")  
openai.api_base = "https://openrouter.ai/api/v1"

if not openai.api_key:
    raise ValueError("❌ A chave de API do OpenRouter não foi definida. Define a variável de ambiente OPENROUTER_API_KEY.")

# Caminho para o ficheiro de estatísticas
TXT_PATH = "output/vehicle_counts.txt"

# === FUNÇÕES ===

def parse_vehicle_stats(txt_path):
    stats = {}
    total = 0

    with open(txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    current_class = None
    for line in lines:
        line = line.strip()

        if line.startswith("Classe:"):
            current_class = line.split(":")[1].strip()
            stats[current_class] = {}
        elif "- Contagem:" in line:
            stats[current_class]["count"] = int(line.split(":")[1].strip())
        elif "- Percentagem:" in line:
            stats[current_class]["percentage"] = float(line.split(":")[1].strip().replace("%", ""))
        elif "- Confiança média:" in line:
            stats[current_class]["avg_conf"] = float(line.split(":")[1].strip())
        elif line.startswith("TOTAL DETETADO:"):
            total = int(line.split(":")[1].strip())

    return stats, total


def generate_prompt(stats, total):
    prompt = "Gere um relatório objetivo e conciso com base nas seguintes estatísticas de contagem de veículos:\n"
    prompt += f"\nTotal de veículos detetados: {total}\n"

    for cls, data in stats.items():
        prompt += (
            f"\nClasse: {cls}\n"
            f" - Contagem: {data['count']}\n"
            f" - Percentagem: {data['percentage']}%\n"
            f" - Confiança média: {data['avg_conf']:.2f}\n"
        )

    prompt += "\nO relatório deve resumir os dados e indicar observações relevantes."
    return prompt


def gerar_relatorio_llm(prompt, modelo="mistralai/mistral-7b-instruct"):
    try:
        response = openai.ChatCompletion.create(
            model=modelo,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=400
        )
        return response["choices"][0]["message"]["content"]
    except openai.error.OpenAIError as e:
        return f"❌ Erro ao gerar relatório com o LLM: {str(e)}"


# === EXECUÇÃO PRINCIPAL ===

if __name__ == "__main__":
    stats, total = parse_vehicle_stats(TXT_PATH)
    prompt = generate_prompt(stats, total)

    print("\n🔹 PROMPT ENVIADO PARA O LLM:\n")
    print(prompt)

    print("\n🔸 RELATÓRIO GERADO PELO LLM:\n")
    relatorio = gerar_relatorio_llm(prompt)
    print(relatorio)

    # Guarda o relatório num ficheiro .txt
    relatorio_path = "output/relatorio_gerado.txt"
    with open(relatorio_path, "w", encoding="utf-8") as f:
        f.write(relatorio)
    print(f"\n✅ Relatório guardado em: {relatorio_path}")
