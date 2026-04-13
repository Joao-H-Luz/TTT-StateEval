import streamlit as st
import random
import pandas as pd
import numpy as np
import joblib
import os

# Configuração da página
st.set_page_config(page_title="Tic Tac Toe com IA (KNN)", layout="centered")

# Função para carregar modelo e scaler
@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load('models/knn.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Erro ao carregar arquivos do modelo: {e}")
        return None, None

model, scaler = load_ml_assets()

# Inicialização do estado da sessão
if 'board' not in st.session_state:
    st.session_state.board = [' ' for _ in range(9)]
if 'current_player' not in st.session_state:
    st.session_state.current_player = 'X'
if 'winner' not in st.session_state:
    st.session_state.winner = None
if 'game_over' not in st.session_state:
    st.session_state.game_over = False
if 'ia_prediction' not in st.session_state:
    st.session_state.ia_prediction = "Aguardando jogada..."
if 'stats' not in st.session_state:
    st.session_state.stats = {"acertos": 0, "erros": 0}

def check_winner_logic(board):
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # linhas
        [0, 3, 6], [1, 4, 7], [2, 5, 8], # colunas
        [0, 4, 8], [2, 4, 6]             # diagonais
    ]
    for condition in win_conditions:
        if board[condition[0]] != ' ' and board[condition[0]] == board[condition[1]] == board[condition[2]]:
            return board[condition[0]]
    if ' ' not in board:
        return 'Empate'
    return None

def board_to_features(board):
    # Mapeamento conforme dataset: X=1, O=-1, Vazio=0
    mapping = {'X': 1, 'O': -1, ' ': 0}
    numeric_board = [mapping[cell] for cell in board]
    
    # Adicionar as 8 pistas de soma e 8 de prioridade conforme notebook (Fase 3)
    combos = [
        [0,1,2], [3,4,5], [6,7,8], # Linhas
        [0,3,6], [1,4,7], [2,5,8], # Colunas
        [0,4,8], [2,4,6]           # Diagonais
    ]
    somas = []
    prioridades = []
    for c in combos:
        soma = numeric_board[c[0]] + numeric_board[c[1]] + numeric_board[c[2]]
        somas.append(soma)
        # Lógica de prioridade: 10 se soma = 3 ou -3, 1 se soma = 2 ou -2, caso contrário 0
        prioridade = 10 if abs(soma) == 3 else (1 if abs(soma) == 2 else 0)
        prioridades.append(prioridade)
    
    features = numeric_board + somas + prioridades
    return np.array(features).reshape(1, -1)

def update_ia_prediction():
    if model and scaler:
        features = board_to_features(st.session_state.board)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        st.session_state.ia_prediction = prediction
        
        # Estado real do tabuleiro para validação
        real_winner = check_winner_logic(st.session_state.board)
        
        # Determinar se a IA acertou ou errou a classificação do estado atual
        is_correct = False
        
        if real_winner is None:
            # Jogo em andamento: IA deve prever 'Tem jogo' ou 'Possibilidade de Fim de Jogo'
            if prediction in ["Tem jogo", "Possibilidade de Fim de Jogo"]:
                is_correct = True
        elif real_winner == 'X':
            if prediction == "X vence":
                is_correct = True
        elif real_winner == 'O':
            if prediction == "O vence":
                is_correct = True
        elif real_winner == 'Empate':
            if prediction == "Empate":
                is_correct = True
        
        # Contabilizar score
        if is_correct:
            st.session_state.stats["acertos"] += 1
        else:
            st.session_state.stats["erros"] += 1

def machine_move():
    if st.session_state.game_over:
        return
        
    empty_spots = [i for i, x in enumerate(st.session_state.board) if x == ' ']
    if empty_spots:
        move = random.choice(empty_spots)
        st.session_state.board[move] = 'O'
        winner = check_winner_logic(st.session_state.board)
        if winner:
            st.session_state.winner = winner
            st.session_state.game_over = True
        else:
            st.session_state.current_player = 'X'
    
    update_ia_prediction()

def human_move(index):
    if st.session_state.board[index] == ' ' and not st.session_state.game_over and st.session_state.current_player == 'X':
        st.session_state.board[index] = 'X'
        winner = check_winner_logic(st.session_state.board)
        if winner:
            st.session_state.winner = winner
            st.session_state.game_over = True
        else:
            st.session_state.current_player = 'O'
            machine_move()
        
        update_ia_prediction()

def reset_game():
    st.session_state.board = [' ' for _ in range(9)]
    st.session_state.current_player = 'X'
    st.session_state.winner = None
    st.session_state.game_over = False
    st.session_state.ia_prediction = "Aguardando jogada..."

# UI do Streamlit
st.title("Jogo da Velha + IA (KNN)")
st.write("Trabalho Prático - Inteligência Artificial")

# Painel Lateral para Scores
with st.sidebar:
    st.header("📊 Performance da IA")
    st.metric("Acertos", st.session_state.stats["acertos"])
    st.metric("Erros", st.session_state.stats["erros"])
    if st.button("Resetar Score"):
        st.session_state.stats = {"acertos": 0, "erros": 0}
        st.rerun()

# Feedback da IA (Requisito do PDF)
st.subheader("🤖 Diagnóstico da IA")
color = "blue"
if "vence" in st.session_state.ia_prediction: color = "green"
elif "Fim" in st.session_state.ia_prediction: color = "orange"
elif "Empate" in st.session_state.ia_prediction: color = "red"

st.markdown(f"**Estado detectado:** :{color}[{st.session_state.ia_prediction}]")

st.markdown("---")

# Estilização do tabuleiro
st.markdown("""
<style>
div.stButton > button:first-child {
    height: 80px;
    width: 100%;
    font-size: 32px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Grid do tabuleiro
cols = st.columns([1, 1, 1])
for i in range(9):
    col = cols[i % 3]
    with col:
        label = st.session_state.board[i] if st.session_state.board[i] != ' ' else "\u200b"
        if st.button(label, key=f"cell_{i}", disabled=st.session_state.board[i] != ' ' or st.session_state.game_over):
            human_move(i)
            st.rerun()

st.markdown("---")

# Controle e Status do Jogo
if st.session_state.game_over:
    if st.session_state.winner == 'Empate':
        st.warning("O jogo terminou em Empate!")
    else:
        st.success(f"O vencedor é: {st.session_state.winner}!")
    
    # Regra do PDF: Se a IA detectar fim de jogo incorretamente, encerra.
    # Como o jogo encerra de qualquer forma quando há vencedor real, vamos apenas mostrar o botão de reset.
    if st.button("Jogar Novamente"):
        reset_game()
        st.rerun()
else:
    st.info(f"Vez do jogador: {st.session_state.current_player}")

# Rodapé informativo
st.caption("A IA analisa o tabuleiro após cada jogada usando um modelo KNN.")
