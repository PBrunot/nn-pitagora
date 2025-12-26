from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(1809)
np.random.seed(1809)

model = Sequential(
    [
        Dense(50, activation="relu", input_dim=2, name="Strato_interno"),
        Dense(1, activation="linear", name="Output_ipotenusa"),
    ]
)

print(model.summary())


def print_model_weights(model):
    for layer in model.layers:
        weights = layer.get_weights()
        print(f"Layer: {layer.name}")
        print("Pesi:", weights[0])  # Matrice pesi
        print("Bias:", weights[1])  # Vettore bias
        print()
        plot_model(
            model, to_file="modello.png", show_shapes=True, show_layer_names=True
        )


print_model_weights(model)

n_samples = 3000  # Numero esempi
cateto1 = np.random.uniform(1, 10, n_samples)  # Cateto A: 1-10
cateto2 = np.random.uniform(1, 10, n_samples)  # Cateto B: 1-10
ipotenusa = np.sqrt(cateto1**2 + cateto2**2)  # Target esatto

X_train = np.column_stack((cateto1, cateto2))  # Input (10000, 2)
y_train = ipotenusa.reshape(-1, 1)  # Output (10000, 1)


def Test_Pitagora(model):
    # Esempio: cateti 3 e 4 → ipotenusa ~5
    input_test = np.array([[3.0, 4.0], [6.0, 8.0]])
    risultati = model.predict(input_test)
    for i, risultato in enumerate(risultati):
        print(f"Ipotenusa di {input_test[i][0]} e {input_test[i][1]}: {risultato[0]:.4f}")
    
    return risultati




model.compile(
    optimizer="adam",  # Ottimizzatore adattivo (buono per iniziare)
    loss="mse",  # Mean Squared Error per regressione
    metrics=["mae"],  # Mean Absolute Error per monitorare
)

history = model.fit(
    X_train,
    y_train,  # Dati generati (1000 esempi)
    epochs=300,  # Iterazioni complete sui dati
    batch_size=32,  # Mini-batch per stabilità
    validation_split=0.2,  # 20% per validazione automatica
    verbose=1,  # Mostra barra progresso
    shuffle=True,  # Mescola dati (riproducibile con seed)
)

Test_Pitagora(model)

print_model_weights(model)

model.save("Rete_Pitagora.keras")
print ("Modello salvato!")

# Dopo training e history = model.fit(...)
fig = plt.figure(figsize=(15, 5))

# 1. Loss per Epoch
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.yscale('log')  # ← Scala log Y per valori piccoli
plt.legend()
plt.grid(True, alpha=0.3)
# limit y-axis to better visualize
plt.ylim(0, 5)

plt.tight_layout()
plt.savefig('rete_35neuroni.png', dpi=200, bbox_inches='tight')
plt.show()
