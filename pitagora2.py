from keras.models import load_model
import numpy as np

model = load_model('Rete_Pitagora.keras')

risultato = model.predict(np.array([[3.0, 4.0]]))
print(f"Ipotenusa calcolata con cateti di 3 e 4: {risultato[0][0]:.4f}")