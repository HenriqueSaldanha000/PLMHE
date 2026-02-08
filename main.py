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
    h_k, alpha, gamma, p_k, beta, q_k, tau, a_k, b_k, mu_a, mu_b, sigma_a, sigma_b = initial_parameters(img)
    
    # Testes    
    print("--- Step 9: Sub-histogram Means ---")
    print(f"Mean Amplitude Lower (mu_a): {mu_a:.4f}")
    print(f"Mean Amplitude Upper (mu_b): {mu_b:.4f}")