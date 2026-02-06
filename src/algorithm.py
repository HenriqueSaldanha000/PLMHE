import numpy as np

def initial_parameters(image):
    """
    Executa os Passos 1 e 2 do algoritmo PLMHE.
    """
    # L é o número total de níveis de cinza possíveis (0 a 256).
    L = 256
    
    # PASSO 1: Cálculo do Histograma
    # image.flatten() transforma a matriz 2D da imagem em um vetor 1D.
    # bins = L define 256 colunas de contagem.
    # range = [0, L] garante que contaremos de 0 até 255.
    # h_k armazenará quantos pixels existem para cada tom de cinza.
    h_k, _ = np.histogram(image.flatten(), bins=L, range=[0, L])
    
    # PASSO 2: Intensidade Média Normalizada (alpha)
    # Se alpha for 0.1, a imagem é muito escura, se for 0.9, é muito clara.
    alpha = np.mean(image) / (L - 1)
    
    # PASSO 3: Cálculo do Expoente Adaptativo (gamma)
    # Este valor determina quão forte será a amplificação inicial do contraste.
    gamma = np.exp(1 - alpha)

    # PASSO 4: Calcula o Power-law Transformation (Pk)
    p_k = np.power(h_k, gamma)
    
    # PASSO 5: Inicialização do parâmetro beta
    # Definido como 0.5 no começo mas é necessario testar outros para achar o ideal.
    beta = 0.5
    
    # PASSO 6: Log Transformation (Qk)
    # Comprime os valores de Pk
    q_k = beta * np.log1p(p_k)
    
    return h_k, alpha, gamma, p_k, beta, q_k