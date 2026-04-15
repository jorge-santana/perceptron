import numpy as np

class Perceptron:
    def __init__(self, taxa_aprendizado = 0.1, epocas = 1000):
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas

    def treinar(self, entradas, rotulos):
        print("Entradas: ", entradas)
        print("Rotulos: ", rotulos)

        num_amostras, num_features = entradas.shape # ele retorna uma tupla com as dimensões do array -> 4 linhas / 2 colunas

        # Parâmetro para o modelo, logo pesos do modelo != da feature peso de cada tupla (dado de entrada)
        # Iniciar os pesos com zero
        self.pesos = np.zeros(num_features)

        print("Pesos iniciais por feature para iniciar o treinamento do modelo: ", self.pesos)

        self.vies = 0 # Inicia o viés do modelo, ele funcionará como um ajuste fino
        print("Viés: ", self.vies)

        for epoca in range(self.epocas):
            print("Epoca: ", epoca)

    def prever(self):
        print("Fazer previsões com base no treinamento")

# [peso (kg), altura (m)]
entradas = np.array([
    [15, 1.0], # criança
    [25, 1.2], # criança
    [65, 1.7], # adulto
    [80, 1.8], # adulto
])

rotulos = np.array([0, 0, 1, 1])

media = entradas.mean(axis=0) # medindo a média por coluna (peso e altura)
desvio_padrao = entradas.std(axis=0) # desvio padrão calculado por colunas
entradas_normalizadas = (entradas - media) / desvio_padrao # normalizar os dados (z-score)

modelo = Perceptron()
modelo.treinar(entradas_normalizadas, rotulos)