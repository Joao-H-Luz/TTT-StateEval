# Tic Tac Toe com Inteligência Artificial (KNN)

Este projeto foi desenvolvido como parte do Trabalho Prático da disciplina de Inteligência Artificial da PUCRS. O objetivo principal é construir um sistema de IA capaz de classificar o estado atual de um tabuleiro de Jogo da Velha utilizando o algoritmo **K-Nearest Neighbors (KNN)**.

## 🎯 Objetivo
Diferente de uma IA que joga, o foco deste trabalho é a **classificação de estados**. A IA recebe o tabuleiro (3x3) e deve identificar se:
- Tem jogo (em andamento)
- Possibilidade de Fim de Jogo (vitoria iminente)
- Empate
- O vence
- X vence

## 🚀 Tecnologias Utilizadas
- **Python 3.x**
- **Streamlit**: Interface web para interação e visualização do jogo.
- **Scikit-learn**: Implementação do modelo KNN, normalização (StandardScaler) e métricas de avaliação.
- **Pandas & Numpy**: Manipulação e processamento de dados.
- **Joblib**: Persistência do modelo treinado e do scaler.
- **Matplotlib & Seaborn**: Geração de gráficos e matriz de confusão nos notebooks.

## 🧠 Desenvolvimento e Engenharia de Atributos (Pistas)
O modelo passou por três fases de evolução, conforme documentado no notebook `KNN.ipynb`:

1.  **Fase 1 (Baseline)**: Treinamento com as 9 posições brutas do tabuleiro.
2.  **Fase 2 (Somas)**: Adição de 8 colunas extras representando a soma de cada combinação de vitória (linhas, colunas e diagonais).
3.  **Fase 3 (Prioridades)**: Refinamento com pesos matemáticos (ex: valor 10 para vitória confirmada e 1 para ameaça de vitória). **Esta versão alcançou ~95-97% de acurácia.**

## 📂 Estrutura do Projeto
- `app/app.py`: Interface Streamlit do jogo.
- `data/`: Datasets originais e processados.
- `models/`: Arquivos binários (`.pkl`) do modelo KNN e do Scaler.
- `notebooks/`: Jupyter Notebooks com a análise exploratória e treinamento do modelo.
- `requirements.txt`: Lista de dependências do projeto.

## 🛠️ Como Executar

### Pré-requisitos
Certifique-se de ter o Python instalado. É recomendado o uso de um ambiente virtual (venv).

### Instalação
1. Clone o repositório ou baixe os arquivos.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

### Executando a Aplicação (Front-end)
Para iniciar o jogo e ver a IA em ação:
```bash
streamlit run app/app.py
```

## 🎮 Como Jogar
1. O usuário joga com o **X** e a máquina (movimentos aleatórios) joga com o **O**.
2. Após cada jogada, a seção **"🤖 Diagnóstico da IA"** mostrará em tempo real qual estado ela detectou no tabuleiro.
3. O painel lateral monitora o **Score da IA** (Acertos vs Erros), validando a predição do modelo contra a lógica real do jogo.

---
**Professor(a):** Silvia Moraes  
**Disciplina:** Inteligência Artificial - PUCRS
