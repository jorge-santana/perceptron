import numpy as np

# Problema: Baseado em peso e altura, quero fazer a predição se é adulto ou criança

class Perceptron:
    def __init__(self, taxa_aprendizado = 0.1, epocas = 1000):
        # taxa de aprendizado
        # muito pequena < 0.5 (aprendizado mais lento, porém mais estável)
        # muito alto > 1 (aprendizado mais rápido, porém pode alucinar)
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        print("Instância da classe Perceptron > Objeto Perceptron")

    def treinar(self):
        print("Treinar o modelo")

    def prever(self):
        print("Fazer previsões com base no treinamento")

modelo = Perceptron()
modelo.treinar()
modelo.prever()

# Dados de entrada (inputs, features) - treinamento / predição
# [peso (kg), altura (m)]
entradas = np.array([
    [15, 1.0], # criança
    [25, 1.2], # criança
    [65, 1.7], # adulto
    [80, 1.8], # adulto
])
# print("Dados de entrada: ", entradas)

# Rótulos (Labels): Treinamento supervisionado
# 0 = criança; 1 = adulto
rotulos = np.array([0, 0, 1, 1])
# print("Dados de validação (treinamento): ", rotulos)
