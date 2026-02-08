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
    
    # PASSO 6: Comprime dos valores de Pk (Qk)
    q_k = beta * np.log1p(p_k)
    
    # PASSO 7: Descobre onde dividir o histograma.
    tau = int(L * alpha)

    # STEP 8: Divide o histograma em 2 subconjuntos baseados no valor de tau
    # Ak: do início até tau
    a_k = q_k[:tau + 1]
    
    # Bk: de tau+1 até o final
    b_k = q_k[tau + 1:]

    # STEP 9: Calcula a media das amplitudes dos 2 dub-histogramas
    # mu_A: média das amplitudes do primeiro sub-histograma
    mu_a = np.mean(a_k)
    
    # mu_B: média das amplitudes do segundo sub-histograma
    mu_b = np.mean(b_k)

    # STEP 10: Calcula o desvio padrão de cada sub-histograma
    # sigma_A: desvio padrão do primeiro sub-histograma
    sigma_a = np.std(a_k)
    
    # sigma_B: desvio padrão do segundo sub-histograma
    sigma_b = np.std(b_k)

    # STEP 11: Soma os sub-histogramas ao seus desvios padrões
    # Dk = Ak + sigma_A
    d_k = a_k + sigma_a
    
    # Ek = Bk + sigma_B
    e_k = b_k + sigma_b
    
    return h_k, alpha, gamma, p_k, beta, q_k, tau, a_k, b_k, mu_a, mu_b, sigma_a, sigma_b, d_k, e_k