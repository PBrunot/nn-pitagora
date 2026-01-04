from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
import numpy as np
import random

random.seed(1809)
np.random.seed(1809)


rete = Sequential(
    [
        Dense(15, activation="relu", input_dim=2, name="Strato_interno"),
        Dense(1, activation="linear", name="Output_ipotenusa"),
    ]
)

numero_campioni = 2000  # Numero esempi
cateti1 = np.random.uniform(0, 15, numero_campioni)  # Cateto A: 0-15
cateti2 = np.random.uniform(0, 15, numero_campioni)  # Cateto B: 0-15
ipotenuse = np.sqrt(cateti1**2 + cateti2**2)  # Target esatto

cateti_training = np.column_stack((cateti1, cateti2))  # Preparara array (10000, 2)
ipotenuse_training = ipotenuse.reshape(-1, 1)  # Prepara output (10000, 1)

rete.compile(loss="mse", optimizer="adam", metrics=["mae"])


# Crea il callback per stampare MAE man mano che impara
class StampaMAECallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            mae_val = logs.get('val_mae', 0)
            print("MAE", mae_val)

callback_mae = StampaMAECallback()

history = rete.fit(
    x=cateti_training,
    y=ipotenuse_training,  # Dati generati
    epochs=100,
    validation_split=0.2,  # 20% per validazione automatica
    shuffle=True,  # Mescola dati
    callbacks=[callback_mae],  # Aggiungi callback
    verbose=0
)

rete.save("Rete_Pitagora.keras")

while True:
    c1 = float(input("Cateto 1="))
    c2 = float(input("Cateto 2="))
    pred = rete.predict(np.array([[c1, c2]]))
    print(f"Ipotenusa calcolata: {pred[0][0]:.4f}")

    # Mostra anche errore di calcolo
    ipotenusa_esatta = np.sqrt(c1**2 + c2**2)
    print(f"Ipotenusa esatta: {ipotenusa_esatta:.4f}")
    print(f"Errore di calcolo: {abs(pred[0][0]-ipotenusa_esatta):.4f} (% di errore = {abs(pred[0][0]-ipotenusa_esatta)/ipotenusa_esatta * 100:0.1f})")
