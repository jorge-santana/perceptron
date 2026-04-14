import numpy as np

class Perceptron:
    def __init__(self, taxa_aprendizado = 0.1, epocas = 1000):
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        print("Instância da classe Perceptron > Objeto Perceptron")

    def treinar(self):
        print("Treinar o modelo")

    def prever(self):
        print("Fazer previsões com base no treinamento")

entradas = np.array([
    [15, 1.0], # criança
    [25, 1.2], # criança
    [65, 1.7], # adulto
    [80, 1.8], # adulto
])

rotulos = np.array([0, 0, 1, 1])

modelo = Perceptron()

# Normalização dos dados de entrada utilizando o metodo z-score

# Media
media = entradas.mean(axis=0) # medindo a média por coluna (peso e altura)
# peso = 15 + 25 + 65 + 80 = 185 / 4 = 46,25
# altura = 1,0 + 1,2 + 1,7 + 1,8 = 5,7 / 4 = 1,425
print(media)

# Desvio padrão
desvio_padrao = entradas.std(axis=0) # desvio padrão calculado por colunas
print(desvio_padrao)

# peso
# (15 - 46,25)**2 = 976,5625
# (25 - 46,25)**2 = 451,5625
# (65 - 46,25)**2 = 351,5625
# (80 - 46,25)**2 = 1.139,0625

# soma = 2.918,75

# Divide pelo número de entradas (4) = 729,6875

# Raiz quadrada: 27,01272848121789

# [15, 1.0], # criança
# [25, 1.2], # criança
# [65, 1.7], # adulto
# [80, 1.8], # adulto

#normalizar os dados (z-score)
desvio_padrao = (entradas - media) / desvio_padrao
print(desvio_padrao)

# [15, 1.0] - [46.25   1.425]
# peso = 15 - 46,25 = -31,25 / 27.01272848 = -1,15686203
# altura = 1.0 - 1.425 = -0,425 / 0.3344772 = -1,27063966
