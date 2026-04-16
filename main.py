import numpy as np
import matplotlib.pyplot as plt

def funcao_heaviside(x):
    return np.where(x >= 0, 1, 0)

class Perceptron:
    def __init__(self, taxa_aprendizado = 0.1, epocas = 100):
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas

    def treinar(self, entradas, rotulos):
        num_amostras, num_features = entradas.shape
        self.pesos = np.zeros(num_features)
        self.vies = 0 # Inicia o viés do modelo, ele funcionará como um ajuste fino

        for epoca in range(self.epocas):
            erros = 0
            print("Época: ", epoca)
            for indice, entrada_individual in enumerate(entradas):
                print("---- Índice: ", indice, "Entrada individual: ", entrada_individual)
                print("---- Pesos: ", self.pesos)
                print("---- Viés: ", self.vies)
                saida_linear = np.dot(entrada_individual, self.pesos) + self.vies # Calcular a saída linear usando os pesos atuais + viés
                # dot -> produto escalar
                # (-1.15686203 * 0) + (-1.27063966 * 0) = 0 + 0 = 0
                print("---- Saída linear: ", saida_linear)

                saida_prevista = funcao_heaviside(saida_linear) # Aplica a função de ativação para obter a classe prevista
                print("---- Ativação: ", saida_prevista)

                ajuste = self.taxa_aprendizado * (rotulos[indice] - saida_prevista)
                # 0.1 * (0 - 1) = -0,1
                print("---- Ajuste: ", ajuste)

                print("---- Erros (antes): ", erros)
                if(ajuste != 0): # Conta quantas vezes o modelo errou (ou precisou corrigir) durante a época
                    erros += 1
                print("---- Erros (depois): ", erros)

                print("---- Pesos (antes): ", self.pesos)
                self.pesos += ajuste * entrada_individual # Ajusta os pesos com base no erro
                #[0. 0.] + (-0.1 * [-1.15686203 -1.27063966]) = [0.1156862  0.12706397]
                print("---- Pesos (depois): ", self.pesos)

                print("---- Vies (antes): ", self.vies)
                self.vies += ajuste
                print("---- Vies (depois): ", self.vies)

            if erros == 0: # Verifica se o treinamento convergiu. Significa que o modelo parou de errar nos dados do treino.
                print(f"---- Treinamento convergiu na época {epoca + 1}")
                break

    def prever(self, entradas):
        saida_linear = np.dot(entradas, self.pesos) + self.vies  # Calcular a saída linear usando os pesos atuais + viés. O dot do numpy realiza o cálculo de produto escalar.
        saida_prevista = funcao_heaviside(saida_linear)  # Aplica a função de ativação para obter a classe prevista
        return saida_prevista

# [peso (kg), altura (m)]
entradas = np.array([
    [15, 1.0], # criança
    [25, 1.2], # criança
    [65, 1.7], # adulto
    [80, 1.8], # adulto
    [50, 1.6], # ambíguo
    [55, 1.6], # ambíguo
    [55, 1.65], # ambíguo
])

# 0 => criança, 1 => adulto
rotulos = np.array([0, 0, 1, 1, 0, 0, 1])

media = entradas.mean(axis=0) # medindo a média por coluna (peso e altura)
desvio_padrao = entradas.std(axis=0) # desvio padrão calculado por colunas
entradas_normalizadas = (entradas - media) / desvio_padrao # normalizar os dados (z-score)

modelo = Perceptron()
modelo.treinar(entradas_normalizadas, rotulos)

# Entrada do novo dado
print("\nDigite um novo dado para classificar:")
peso_input = float(input("Peso (kg). Ex 75:"))
altura_input = float(input("Altura (m). Ex 1.85:"))

novo_dado = np.array([[peso_input, altura_input]])

# Normalização
novo_dado_normalizado = (novo_dado - media) / desvio_padrao # normalizar os dados (z-score)

# Previsão
previsao = modelo.prever(novo_dado_normalizado)[0]
print("Previsão: ", previsao)

classe_prevista = "Adulto" if previsao == 1 else "Criança"
print("Resultado (predição): ", classe_prevista)

## -----------------------

# =========================
# Visualização
# =========================
plt.figure()

# Plot dos dados de treino com rótulos
for i in range(len(entradas)):
    peso, altura = entradas[i]
    classe = "Adulto" if rotulos[i] == 1 else "Criança"
    cor = "orange" if rotulos[i] == 1 else "blue"

    plt.scatter(
        entradas_normalizadas[i, 0],
        entradas_normalizadas[i, 1],
        color=cor,
        s=100
    )

    plt.annotate(
        f"{classe}\n({peso}kg, {altura}m)",
        (entradas_normalizadas[i, 0], entradas_normalizadas[i, 1]),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=9
    )

# =========================
# Linha de decisão
# =========================
pesos = modelo.pesos
vies = modelo.vies

valores_x = np.linspace(
    entradas_normalizadas[:, 0].min(),
    entradas_normalizadas[:, 0].max(),
    100
)

if pesos[1] != 0:
    valores_y = -(pesos[0] * valores_x + vies) / pesos[1]
    plt.plot(valores_x, valores_y, label="Linha de decisão")

# =========================
# Novo dado destacado
# =========================
plt.scatter(
    novo_dado_normalizado[:, 0],
    novo_dado_normalizado[:, 1],
    marker="*",
    s=250,
    label="Novo dado"
)

plt.annotate(
    f"{classe_prevista}\n({peso_input}kg, {altura_input}m)",
    (novo_dado_normalizado[0, 0], novo_dado_normalizado[0, 1]),
    textcoords="offset points",
    xytext=(5, 5),
    fontsize=10,
    fontweight="bold"
)

# =========================
# Ajustes finais
# =========================
plt.xlabel("Peso (normalizado)")
plt.ylabel("Altura (normalizada)")
plt.title("Perceptron - Classificação: Criança vs Adulto")
plt.legend()
plt.grid()

plt.show()