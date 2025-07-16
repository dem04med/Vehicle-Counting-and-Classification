def detectar_anomalias_video_unico(stats, total_veiculos, limiar_conf=0.3, max_deteccao=80):
    """
    Deteta anomalias num único vídeo com base em regras heurísticas simples.

    :param stats: Dicionário com estatísticas por classe. Formato:
                  {
                      "car": {"count": 25, "avg_conf": 0.78},
                      "truck": {"count": 5, "avg_conf": 0.55},
                      ...
                  }
    :param total_veiculos: Total de veículos detetados no vídeo
    :param limiar_conf: Confiança média mínima aceitável por classe
    :param max_deteccao: Número máximo razoável de deteções por classe
    :return: Lista de anomalias detetadas (strings)
    """
    anomalias = []

    # Anomalia 1: nenhum veículo detetado
    if total_veiculos == 0:
        anomalias.append("❗ Nenhum veículo foi detetado no vídeo.")
        return anomalias  # não faz sentido verificar mais neste caso

    # Anomalia 8: total de classes muito reduzido
    if len(stats) < 2:
        anomalias.append(f"⚠️ Apenas {len(stats)} classe(s) de veículos foram detetadas — diversidade muito reduzida.")

    # Anomalias por classe
    for classe, info in stats.items():
        count = info.get("count", 0)
        conf = info.get("avg_conf", 0)
        percentage = (count / total_veiculos) * 100 if total_veiculos > 0 else 0

        # Anomalia 2: confiança média muito baixa
        if conf < limiar_conf:
            anomalias.append(f"⚠️ Confiança média baixa na classe '{classe}': {conf:.2f}")

        # Anomalia 3: deteções excessivas (possível falso positivo)
        if count > max_deteccao:
            anomalias.append(f"❗ Contagem elevada para '{classe}': {count} deteções")

        # Anomalia 4: classe desconhecida
        if classe.lower() in {"unknown", "desconhecido", "outro"}:
            anomalias.append("⚠️ Classe desconhecida detetada: 'unknown'")

        # Anomalia 5: classe representa percentagem excessiva do total
        if percentage > 85:
            anomalias.append(f"⚠️ A classe '{classe}' representa {percentage:.1f}% dos veículos — possível concentração anómala.")

        # Anomalia 6: classe representa percentagem muito reduzida
        if 0 < percentage < 3:
            anomalias.append(f"⚠️ A classe '{classe}' representa apenas {percentage:.1f}% — pode estar sub-representada.")

        # Anomalia 7: confiança muito baixa apesar de deteções
        if count > 0 and conf < 0.3:
            anomalias.append(f"❗ Classe '{classe}' detetada com confiança muito baixa ({conf:.2f}) — possíveis falsos positivos.")

        # Anomalia 9: classe com count = 0 mas presente no relatório
        if count == 0:
            anomalias.append(f"⚠️ Classe '{classe}' está no relatório mas com 0 deteções — verificar se deve ser incluída.")

    return anomalias


# Exemplo de uso local para testes
if __name__ == "__main__":
    exemplo_stats = {
        "car": {"count": 120, "avg_conf": 0.85},
        "truck": {"count": 2, "avg_conf": 0.25},
        "unknown": {"count": 1, "avg_conf": 0.60},
        "bus": {"count": 0, "avg_conf": 0.90}
    }
    total = sum([v["count"] for v in exemplo_stats.values()])

    resultado = detectar_anomalias_video_unico(exemplo_stats, total)
    if not resultado:
        print("[INFO] Nenhuma anomalia detetada.")
    else:
        print("[INFO] Anomalias encontradas:")
        for r in resultado:
            print(" -", r)
