# 🌦️ Meteograma

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

> Script em Python para download de dados meteorológicos do **Global Forecast System (GFS)** via Open-Meteo API, com processamento de dados e geração de visualizações.

## 🚀 Funcionalidades

- **Download Automático**: Coleta dados horários de previsão para 7 dias.
- **Variáveis Abrangentes**: Temperatura, Umidade, Precipitação, Vento (velocidade/direção/rajadas), Cobertura de Nuvens (vários níveis) e mais.
- **Cálculos Derivados**:
  - Ponto de Orvalho (Dewpoint).
  - Componentes U e V do vento.
  - Simulação estimada de altura de ondas e maré (baseado em vento e harmônicos simples).
- **Exportação**: Salva os dados brutos em JSON e processados em CSV.

## 🛠️ Tecnologias Utilizadas

- **[Open-Meteo API](https://open-meteo.com/)**: Fonte dos dados meteorológicos.
- **Pandas**: Manipulação e análise de dados tabulares.
- **NumPy**: Cálculos matemáticos e vetoriais.
- **HTTPX**: Cliente HTTP moderno e rápido para requisições.

## 📦 Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/leff22/Meteograma.git
   cd Meteograma
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Como Usar

Execute o script principal para baixar os dados da região configurada (padrão: São Carlos/SP):

```bash
python download_dados.py
```

Os arquivos serão gerados na pasta `data/`:
- `gfs_sao_carlos_raw.json`: Dados brutos da API.
- `gfs_sao_carlos_hourly.csv`: Tabela formatada com todas as variáveis.
- **Altere os caminhos e as pastas** no script conforme necessário.
- Altere também as coordenadas e o período de tempo desejado.

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---
Desenvolvido por [Leandro Faria](https://github.com/leff22)