import cv2
import numpy as np
import os

from src.algorithm import initial_parameters

# Caminho da imagem de teste.
input_path = "data/input/teste.jpg" 

img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

# Verificação
if img is None:
    print(f"Erro: Não foi possível encontrar a imagem")
else:
    # Chama a função.
    h_k, alpha, gamma, p_k, beta, q_k = initial_parameters(img)
    
    # Testes    
    print("--- PLMHE: Initial Steps Success ---")
    print(f"Alpha: {alpha:.4f}")
    print(f"Gamma: {gamma:.4f}")
    print(f"Max Qk (Log compressed): {np.max(q_k):.2f}")