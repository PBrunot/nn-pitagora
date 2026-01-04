from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import random
import os

random.seed(1809)
np.random.seed(1809)

rete = Sequential(
    [
        Dense(15, activation="relu", input_dim=2, name="Strato_interno"),
        Dense(1, activation="linear", name="Output_ipotenusa"),
    ]
)

print(rete.summary())


def print_model_weights(model):
    """Stampa i pesi del modello in un'unica tabella con tutti gli strati."""

    # Estrai pesi da entrambi i layer
    layer0_weights = model.layers[0].get_weights()
    layer1_weights = model.layers[1].get_weights()

    W0 = layer0_weights[0]  # (2, 15) - pesi input -> hidden
    b0 = layer0_weights[1]  # (15,) - bias hidden layer
    W1 = layer1_weights[0]  # (15, 1) - pesi hidden -> output
    b1 = layer1_weights[1]  # (1,) - bias output layer

    print(f"\n{'='*120}")
    print(f"PESI E BIAS DELLA RETE NEURALE")
    print(f"{'='*120}")
    print(f"\nArchitettura: Input(2) → Hidden({W0.shape[1]}) → Output(1)")
    print(f"Forme: W_hidden{W0.shape}, b_hidden{b0.shape}, W_output{W1.shape}, b_output{b1.shape}")

    # Intestazione tabella
    print(f"\n{'Neurone':<10} {'weight_i1':<15} {'weight_i2':<15} {'bias_interno':<15} {'weight_output':<15} {'bias_output':<15}")
    print("-" * 95)

    # Stampa righe (15 neuroni dello strato nascosto)
    for i in range(W0.shape[1]):
        weight_i1 = W0[0, i]
        weight_i2 = W0[1, i]
        bias_hidden = b0[i]
        weight_output = W1[i, 0]
        bias_out = b1[0] if i == 0 else ""  # Mostra bias output solo sulla prima riga

        if i == 0:
            print(f"{i:<10} {weight_i1:< 15.6f} {weight_i2:< 15.6f} {bias_hidden:< 15.6f} {weight_output:< 15.6f} {bias_out:< 15.6f}")
        else:
            print(f"{i:<10} {weight_i1:< 15.6f} {weight_i2:< 15.6f} {bias_hidden:< 15.6f} {weight_output:< 15.6f}")

    print(f"{'='*120}\n")

    # Genera il diagramma del modello
    plot_model(
        model, to_file="modello.png", show_shapes=True, show_layer_names=True
    )
    print(f"Diagramma modello salvato in 'modello.png'")


print_model_weights(rete)


# Callback per salvare il modello ad ogni epoca
class SalvaAdOgniEpoca(Callback):
    def __init__(self, cartella="saved_models"):
        super().__init__()
        self.cartella = cartella
        # Crea la cartella se non esiste
        if not os.path.exists(cartella):
            os.makedirs(cartella)
            print(f"Cartella '{cartella}' creata per salvare i modelli.")

    def on_epoch_end(self, epoch, logs=None):
        # Salva il modello alla fine di ogni epoca (epoch parte da 0)
        nome_file = os.path.join(self.cartella, f"model_epoch_{epoch+1:03d}.keras")
        self.model.save(nome_file)
        print(f"\n→ Modello salvato: {nome_file}")


def Test_Pitagora(model):
    # Esempio: cateti 3 e 4 → ipotenusa ~5
    input_test = np.array([[3.0, 4.0], [6.0, 8.0]])
    risultati = model.predict(input_test)
    for i, risultato in enumerate(risultati):
        print(
            f"Ipotenusa di {input_test[i][0]} e {input_test[i][1]}: {risultato[0]:.4f}"
        )

    return risultati


numero_campioni = 2000  # Numero esempi
cateti1 = np.random.uniform(0, 15, numero_campioni)  # Cateto A: 0-15
cateti2 = np.random.uniform(0, 15, numero_campioni)  # Cateto B: 0-15
ipotenuse = np.sqrt(cateti1**2 + cateti2**2)  # Target esatto

cateti_training = np.column_stack((cateti1, cateti2))  # Preparara array (10000, 2)
ipotenuse_training = ipotenuse.reshape(-1, 1)  # Prepara output (10000, 1)

rete.compile(loss="mse", optimizer="adam", metrics=["mae"])

# Crea il callback per salvare ad ogni epoca
salva_callback = SalvaAdOgniEpoca(cartella="saved_models")

history = rete.fit(
    x=cateti_training,
    y=ipotenuse_training,  # Dati generati
    epochs=250,
    validation_split=0.2,  # 20% per validazione automatica
    shuffle=True,  # Mescola dati
    callbacks=[salva_callback],  # Aggiungi il callback
)

# Valore atteso 5
print(rete.predict(np.array([[3,4]])))

Test_Pitagora(rete)

print_model_weights(rete)

rete.save("Rete_Pitagora.keras")
print("Modello salvato!")

# Dopo training e history = model.fit(...)
fig = plt.figure(figsize=(15, 5))

# 1. Loss per Epoch
plt.plot(history.history["val_mae"], label="Validazione MAE", linewidth=2)
plt.title("Andamento allenamento")
plt.xlabel("Epoca")
plt.ylabel("Errore Assoluto Medio (MAE)")
plt.yscale("log")  # Logaritmo sulla y-axis
plt.legend()
plt.grid(True, alpha=0.3)
# limit y-axis to better visualize
plt.ylim(0, 5)
plt.tight_layout()
plt.show()
