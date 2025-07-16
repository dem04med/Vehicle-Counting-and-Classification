from dotenv import load_dotenv
import openai
import os
from fpdf import FPDF
from analise_anomalias import detectar_anomalias_video_unico  # <- NOVO

# === CONFIGURAÇÕES ===

load_dotenv()

openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

if not openai.api_key:
    raise ValueError("❌ A chave de API do OpenRouter não foi definida. Define a variável de ambiente OPENROUTER_API_KEY.")

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


def formatar_anomalias(anomalias):
    if not anomalias:
        return "Nenhuma anomalia foi detetada com base nas estatísticas deste vídeo."

    texto = "As seguintes anomalias foram detetadas na análise local do vídeo:\n"
    for a in anomalias:
        texto += f"- {a}\n"
    return texto


def generate_prompt(stats, total, anomalias_texto=""):
    prompt = (
        "Gere um relatório técnico e estruturado com base nas seguintes estatísticas de contagem de veículos.\n"
        "O relatório deve seguir este formato e estrutura obrigatórios:\n\n"

        "=== Estatísticas Gerais ===\n"
        "Inclua o total de veículos detetados.\n\n"

        "=== Análise por Classe ===\n"
        "Para cada classe, apresente de forma clara:\n"
        "- Nome da classe\n"
        "- Número de veículos detetados\n"
        "- Percentagem em relação ao total\n"
        "- Confiança média (com duas casas decimais)\n\n"

        "=== Anomalias Detetadas ===\n"
        "Liste cada anomalia detetada com bullet points (•).\n"
        "Se não existirem anomalias, escreva: 'Nenhuma anomalia foi detetada.'\n\n"

        "Evite cortar informação. Use linhas completas. Use linguagem clara e objetiva.\n"
        "Evite frases demasiado longas e não omita dados relevantes.\n\n"

        "=== Dados para Análise ===\n"
        f"Total de veículos detetados: {total}\n"
    )

    for cls, data in stats.items():
        prompt += (
            f"\nClasse: {cls}\n"
            f"- Contagem: {data['count']}\n"
            f"- Percentagem: {data['percentage']}%\n"
            f"- Confiança média: {data['avg_conf']:.2f}\n"
        )

    if anomalias_texto:
        prompt += "\n\n=== Análise de Anomalias ===\n"
        prompt += anomalias_texto

    prompt += "\n\nGere o relatório seguindo rigorosamente o formato acima."
    return prompt


def gerar_relatorio_llm(prompt, modelo="mistralai/mistral-7b-instruct"):
    try:
        response = openai.ChatCompletion.create(
            model=modelo,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=1500
        )
        return response["choices"][0]["message"]["content"]
    except openai.error.OpenAIError as e:
        return f"❌ Erro ao gerar relatório com o LLM: {str(e)}"


def limpar_caracteres(texto):
    """Remove ou substitui caracteres incompatíveis com a codificação 'latin-1' usada pelo FPDF."""
    substituicoes = {
        "❗": "[!]",
        "⚠️": "[Atenção]",
        "→": "->",
        "–": "-",
        "—": "-",
        "’": "'",
        "“": '"',
        "”": '"',
        "•": "-",
        "✔": "[ok]",
        "✖": "[x]",
        "🛈": "[info]",
        "➡": "->"
    }
    for char, substituto in substituicoes.items():
        texto = texto.replace(char, substituto)
    return texto


def gerar_pdf(relatorio_texto, caminho_pdf):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    # Limpar caracteres antes de gerar PDF
    relatorio_texto = limpar_caracteres(relatorio_texto)

    for linha in relatorio_texto.split("\n"):
        pdf.multi_cell(0, 10, txt=linha)

    pdf.output(caminho_pdf)


# === EXECUÇÃO PRINCIPAL ===

def main():
    stats, total = parse_vehicle_stats(TXT_PATH)

    # Nova análise de anomalias (sem histórico)
    anomalias = detectar_anomalias_video_unico(stats, total)
    texto_anomalias = formatar_anomalias(anomalias)

    # Geração do prompt
    prompt = generate_prompt(stats, total, texto_anomalias)

    print("\n🔹 PROMPT ENVIADO PARA O LLM:\n")
    print(prompt)

    print("\n🔸 RELATÓRIO GERADO PELO LLM:\n")
    relatorio = gerar_relatorio_llm(prompt)
    print(relatorio)

    # Guardar como .txt
    relatorio_txt_path = "output/relatorio_gerado.txt"
    with open(relatorio_txt_path, "w", encoding='utf-8') as f:
        f.write(relatorio)
    print(f"\n✅ Relatório de texto guardado em: {relatorio_txt_path}")

    # Guardar como PDF
    relatorio_pdf_path = "output/relatorio_gerado.pdf"
    gerar_pdf(relatorio, relatorio_pdf_path)
    print(f"\n✅ Relatório PDF guardado em: {relatorio_pdf_path}")


if __name__ == "__main__":
    main()
