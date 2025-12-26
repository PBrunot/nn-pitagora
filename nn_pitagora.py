import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, Adagrad, SGD
from keras.callbacks import Callback
from keras.models import load_model
import pickle
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def imposta_seme_casuale(seme=42):
    """Imposta il seme casuale per la riproducibilità."""
    np.random.seed(seme)


def genera_dati_addestramento(n_campioni=5000, range_min=-5, range_max=5):
    """Genera dati di addestramento per la rete neurale del teorema di Pitagora.

    Args:
        n_campioni (int): Numero di campioni da generare
        range_min (float): Valore minimo per a e b
        range_max (float): Valore massimo per a e b

    Returns:
        tuple: (X_addestramento, y_addestramento) dove X_addestramento sono i dati di input e y_addestramento il target
    """
    a_addestramento = np.random.uniform(range_min, range_max, n_campioni)
    b_addestramento = np.random.uniform(range_min, range_max, n_campioni)
    y_addestramento = np.sqrt(a_addestramento**2 + b_addestramento**2)
    X_addestramento = np.column_stack((a_addestramento, b_addestramento))
    return X_addestramento, y_addestramento


def costruisci_modello(
    unita_nascoste=50, attivazione="tanh", tasso_apprendimento=0.001
):
    """Costruisce e compila il modello di rete neurale.

    Args:
        unita_nascoste (int): Numero di neuroni nello strato nascosto
        attivazione (str): Funzione di attivazione per lo strato nascosto
        tasso_apprendimento (float): Tasso di apprendimento per l'ottimizzatore

    Returns:
        Sequential: Modello Keras compilato
    """
    modello = Sequential(
        [
            Dense(unita_nascoste, activation=attivazione, input_shape=(2,)),
            Dense(1, activation="linear"),
        ]
    )

    modello.compile(
        optimizer=SGD(learning_rate=tasso_apprendimento), 
        loss="mse", 
        metrics=["mae"],

    )

    return modello


def mostra_info_modello(modello):
    """Mostra informazioni sull'architettura del modello."""
    print("Architettura del Modello:")
    modello.summary()


class SalvaModelloAEpoche(Callback):
    """Callback personalizzato per salvare i pesi del modello in epoche specifiche."""

    def __init__(self, epoche_da_salvare, salva_modelli_completi=True):
        super().__init__()
        self.epoche_da_salvare = epoche_da_salvare
        self.modelli_salvati = {}
        self.salva_modelli_completi = salva_modelli_completi
        
        # Crea directory per modelli salvati se necessario
        if self.salva_modelli_completi:
            os.makedirs("saved_models", exist_ok=True)

    def on_epoch_end(self, epoca, logs=None):
        numero_epoca = epoca + 1
        if numero_epoca in self.epoche_da_salvare:
            print(f"\nSalvataggio modello all'epoca {numero_epoca}")
            
            # Salva i pesi per compatibilità con codice esistente
            self.modelli_salvati[numero_epoca] = [
                p.copy() for p in self.model.get_weights()
            ]
            
            # Salva modello completo usando model.save()
            if self.salva_modelli_completi:
                model_path = f"saved_models/model_epoch_{numero_epoca:03d}.keras"
                self.model.save(model_path)
                print(f"Modello completo salvato in '{model_path}'")


def addestra_modello(
    model,
    X_train,
    y_train,
    epochs=75,
    batch_size=32,
    validation_split=0.2,
    epochs_to_save=None,
    save_complete_models=True,
):
    """Train the neural network model.

    Args:
        model: Compiled Keras model
        X_train: Training input data
        y_train: Training target data
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        validation_split (float): Fraction of data to use for validation
        epochs_to_save (list): Epochs at which to save model weights
        save_complete_models (bool): Whether to save complete models using model.save()

    Returns:
        tuple: (history, save_callback) Training history and callback with saved models
    """
    print("\nTraining the model...")

    if epochs_to_save is None:
        epochs_to_save = [1, 3, 5, 10, 20, 30, 40, 50, 75]

    save_callback = SalvaModelloAEpoche(epochs_to_save, save_complete_models)

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
        callbacks=[save_callback],
    )

    return history, save_callback


def save_model_weights(model, filename="model_weights.pkl"):
    """Save model weights for external use (e.g., Manim animation).

    Args:
        model: Trained Keras model
        filename (str): Output filename for weights
    """
    weights = model.get_weights()
    weights_data = {
        "W1": weights[0],  # Input to hidden layer weights
        "b1": weights[1],  # Hidden layer biases
        "W2": weights[2],  # Hidden to output layer weights
        "b2": weights[3],  # Output layer bias
    }
    with open(filename, "wb") as f:
        pickle.dump(weights_data, f)
    print(f"\nModel weights saved to '{filename}'")


def load_saved_model(epoch_number, models_directory="saved_models"):
    """Carica un modello salvato da una specifica epoca.
    
    Args:
        epoch_number (int): Numero dell'epoca del modello da caricare
        models_directory (str): Directory contenente i modelli salvati
        
    Returns:
        keras.Model: Modello caricato
    """
    model_path = os.path.join(models_directory, f"model_epoch_{epoch_number:03d}.keras")
    
    if not os.path.exists(model_path):
        available_models = []
        if os.path.exists(models_directory):
            for file in os.listdir(models_directory):
                if file.startswith("model_epoch_"):
                    available_models.append(file)
        
        raise FileNotFoundError(
            f"Modello per epoca {epoch_number} non trovato in '{model_path}'. "
            f"Modelli disponibili: {available_models}"
        )
    
    model = load_model(model_path)
    print(f"Modello epoca {epoch_number} caricato da '{model_path}'")
    return model


def grafico_cronologia_addestramento(
    cronologia, nome_file_salvataggio="perdita_addestramento.png", mostra_grafico=True
):
    """Grafico della perdita e cronologia MAE dell'addestramento.

    Args:
        cronologia: Oggetto cronologia addestramento Keras
        nome_file_salvataggio (str): Nome file per salvare il grafico
        mostra_grafico (bool): Se mostrare il grafico
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(cronologia.history["loss"], label="Perdita Addestramento")
    plt.plot(cronologia.history["val_loss"], label="Perdita Validazione")
    plt.xlabel("Epoca")
    plt.ylabel("Perdita (MSE)")
    plt.title("Perdita Durante Addestramento")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(cronologia.history["mae"], label="MAE Addestramento")
    plt.plot(cronologia.history["val_mae"], label="MAE Validazione")
    plt.xlabel("Epoca")
    plt.ylabel("Errore Assoluto Medio")
    plt.title("MAE Durante Addestramento")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(nome_file_salvataggio)
    print(f"\nGrafici addestramento salvati come '{nome_file_salvataggio}'")

    if mostra_grafico:
        plt.show()


def valuta_modello_su_casi_test(modello):
    """Valuta il modello su casi di test specifici.

    Args:
        modello: Modello Keras addestrato
    """
    print("\n" + "=" * 60)
    print("Test del modello su coppie specifiche (a, b):")
    print("=" * 60)

    casi_test = [
        (1, 2),  # Dovrebbe essere sqrt(5) ≈ 2.236
        (3, 4),  # Dovrebbe essere 5
        (-3, 4),  # Dovrebbe essere 5
        (3, -4),  # Dovrebbe essere 5
        (-3, -4),  # Dovrebbe essere 5
        (1, 1),  # Dovrebbe essere sqrt(2) ≈ 1.414
        (-1, -1),  # Dovrebbe essere sqrt(2) ≈ 1.414
        (0, 5),  # Dovrebbe essere 5
        (5, 0),  # Dovrebbe essere 5
        (-5, 0),  # Dovrebbe essere 5
        (0, -5),  # Dovrebbe essere 5
        (4.5, 4.5),  # Dovrebbe essere sqrt(40.5) ≈ 6.364
    ]

    for a, b in casi_test:
        X_test = np.array([[a, b]])
        predizione = modello.predict(X_test, verbose=0)[0][0]
        valore_reale = np.sqrt(a**2 + b**2)
        errore = abs(predizione - valore_reale)
        errore_pct = (errore / valore_reale) * 100 if valore_reale != 0 else 0

        print(
            f"a={a:6.2f}, b={b:6.2f} | "
            f"Predetto: {predizione:7.4f} | "
            f"Reale: {valore_reale:7.4f} | "
            f"Errore: {errore:6.4f} ({errore_pct:5.2f}%)"
        )

    print("=" * 60)


def ottieni_predizioni_per_epoca(
    pesi, X_input, forma_output, unita_nascoste=50, attivazione="tanh"
):
    """Crea un modello temporaneo e ottiene predizioni per un'epoca specifica.

    Args:
        pesi: Pesi del modello da epoca specifica
        X_input: Dati di input per le predizioni
        forma_output: Forma per rimodellare le predizioni
        unita_nascoste (int): Numero di unità nascoste
        attivazione (str): Funzione di attivazione

    Returns:
        ndarray: Predizioni rimodellate
    """
    modello_temp = Sequential(
        [
            Dense(unita_nascoste, activation=attivazione, input_shape=(2,)),
            Dense(1, activation="linear"),
        ]
    )
    modello_temp.build((None, 2))
    modello_temp.set_weights(pesi)
    return modello_temp.predict(X_input, verbose=0).reshape(forma_output)


def ottieni_colore_epoca(indice_epoca, totale_epoche):
    """Genera gradiente di colore dal rosso al verde.

    Args:
        indice_epoca (int): Indice dell'epoca corrente
        totale_epoche (int): Numero totale di epoche

    Returns:
        str: Nome del colore per l'epoca
    """
    colori = ["red", "orange", "yellow", "yellowgreen", "green"]
    posizione = indice_epoca / (totale_epoche - 1) if totale_epoche > 1 else 0
    indice_colore = posizione * (len(colori) - 1)
    indice_basso = int(indice_colore)
    return colori[indice_basso]


def crea_visualizzazione_3d(
    callback_salvataggio,
    epoche_da_graficare,
    risoluzione=50,
    unita_nascoste=50,
    attivazione="tanh",
):
    """Crea visualizzazione 3D confrontando funzione reale con predizioni NN.

    Args:
        callback_salvataggio: Oggetto callback con pesi modello salvati
        epoche_da_graficare (list): Lista di epoche da visualizzare
        risoluzione (int): Risoluzione griglia per visualizzazione
        unita_nascoste (int): Numero di unità nascoste nel modello
        attivazione (str): Funzione di attivazione usata nel modello

    Returns:
        tuple: (fig, range_a, range_b, Z_reale, X_griglia)
    """
    print("\nGenerazione grafico confronto 3D...")

    # Crea meshgrid
    range_a = np.linspace(-5, 5, risoluzione)
    range_b = np.linspace(-5, 5, risoluzione)
    A, B = np.meshgrid(range_a, range_b)

    # Calcola valori funzione reale
    Z_reale = np.sqrt(A**2 + B**2)

    # Prepara input per predizioni
    X_griglia = np.column_stack((A.ravel(), B.ravel()))

    # Crea figura
    fig = go.Figure()

    print("\nQualità Approssimazione a epoche diverse:")
    print("=" * 60)

    # Aggiungi superficie funzione reale
    fig.add_trace(
        go.Surface(
            x=A,
            y=B,
            z=Z_reale,
            colorscale=[[0, "blue"], [1, "blue"]],
            opacity=0.3,
            showscale=False,
            name="Funzione Reale",
            hovertemplate="a: %{x}<br>b: %{y}<br>Reale: %{z:.2f}<extra></extra>",
        )
    )

    # Aggiungi superfici predizioni NN per ogni epoca
    for indice, epoca in enumerate(epoche_da_graficare):
        Z_predetto = ottieni_predizioni_per_epoca(
            callback_salvataggio.modelli_salvati[epoca],
            X_griglia,
            A.shape,
            unita_nascoste,
            attivazione,
        )

        # Calcola metriche
        differenza = np.abs(Z_predetto - Z_reale)
        errore_medio = np.mean(differenza)
        errore_max = np.max(differenza)
        rmse = np.sqrt(np.mean(differenza**2))

        print(
            f"Epoca {epoca:2d} | MAE: {errore_medio:7.4f} | Errore Max: {errore_max:7.4f} | RMSE: {rmse:7.4f}"
        )

        colore = ottieni_colore_epoca(indice, len(epoche_da_graficare))
        fig.add_trace(
            go.Surface(
                x=A,
                y=B,
                z=Z_predetto,
                colorscale=[[0, colore], [1, colore]],
                opacity=0.4,
                showscale=False,
                name=f"Epoca {epoca}",
                hovertemplate=f"Epoca {epoca}<br>a: %{{x}}<br>b: %{{y}}<br>Predetto: %{{z:.2f}}<extra></extra>",
            )
        )

    print("=" * 60)

    # Aggiorna layout
    range_epoche = f"Epoche {epoche_da_graficare[0]}→{epoche_da_graficare[-1]}"
    fig.update_layout(
        title=f"Progresso Apprendimento Rete Neurale: Approssimazione f(a,b) = √(a² + b²)<br><sub>Blu=Reale, Rosso→Verde={range_epoche}</sub>",
        scene=dict(
            xaxis_title="a",
            yaxis_title="b",
            zaxis_title="f(a,b)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
        ),
        height=800,
        width=1200,
        showlegend=True,
        legend=dict(x=0.7, y=0.9),
    )

    return fig, range_a, range_b, Z_reale, X_griglia


def salva_dati_epoca_per_animazione(
    callback_salvataggio,
    epoche_da_salvare,
    range_a,
    range_b,
    Z_reale,
    X_griglia,
    unita_nascoste=50,
    attivazione="tanh",
    nome_file="superfici_epoche.pkl",
):
    """Salva dati epoca per animazione Manim.

    Args:
        callback_salvataggio: Oggetto callback con pesi modello salvati
        epoche_da_salvare (list): Lista di epoche da salvare
        range_a, range_b: Range di coordinate
        Z_reale: Valori funzione reale
        X_griglia: Griglia input per predizioni
        unita_nascoste (int): Numero di unità nascoste
        attivazione (str): Funzione di attivazione
        nome_file (str): Nome file di output
    """
    print("\nSalvataggio dati epoca per animazione Manim...")
    dati_epoca = {
        "a_range": range_a,
        "b_range": range_b,
        "Z_actual": Z_reale,
        "epochs": {},
    }

    for epoca in epoche_da_salvare:
        Z_predetto = ottieni_predizioni_per_epoca(
            callback_salvataggio.modelli_salvati[epoca],
            X_griglia,
            Z_reale.shape,
            unita_nascoste,
            attivazione,
        )
        dati_epoca["epochs"][epoca] = Z_predetto

    with open(nome_file, "wb") as f:
        pickle.dump(dati_epoca, f)
    print(f"Dati superficie epoca salvati in '{nome_file}' per animazione Manim")


def salva_grafico_interattivo(fig, nome_file="confronto_3d_epoche.html"):
    """Salva grafico 3D interattivo come file HTML.

    Args:
        fig: Oggetto figura Plotly
        nome_file (str): Nome file di output
    """
    fig.write_html(nome_file)
    print(f"\nGrafico confronto 3D interattivo salvato come '{nome_file}'")


def principale():
    """Funzione principale per orchestrare l'intero flusso di lavoro della rete neurale."""
    # Configurazione
    unita_nascoste = 20
    attivazione = "relu"
    tasso_apprendimento = 0.001
    epoche = 75
    epoche_da_salvare = [1, 3, 5, 10, 20, 30, 40, 50, 75]

    # Imposta seme casuale
    imposta_seme_casuale(42)

    # Genera dati di addestramento
    X_addestramento, y_addestramento = genera_dati_addestramento(n_campioni=5000)

    # Costruisci e mostra modello
    modello = costruisci_modello(unita_nascoste, attivazione, tasso_apprendimento)
    mostra_info_modello(modello)

    # Addestra il modello
    cronologia, callback_salvataggio = addestra_modello(
        modello,
        X_addestramento,
        y_addestramento,
        epochs=epoche,
        epochs_to_save=epoche_da_salvare,
    )

    # Salva pesi del modello
    save_model_weights(modello)

    # Grafico cronologia addestramento
    grafico_cronologia_addestramento(cronologia)

    # Valuta modello su casi test
    valuta_modello_su_casi_test(modello)

    # Crea visualizzazione 3D
    fig, range_a, range_b, Z_reale, X_griglia = crea_visualizzazione_3d(
        callback_salvataggio,
        epoche_da_salvare,
        unita_nascoste=unita_nascoste,
        attivazione=attivazione,
    )

    # Salva dati epoca per animazione
    salva_dati_epoca_per_animazione(
        callback_salvataggio,
        epoche_da_salvare,
        range_a,
        range_b,
        Z_reale,
        X_griglia,
        unita_nascoste=unita_nascoste,
        attivazione=attivazione,
    )

    # Salva grafico interattivo
    salva_grafico_interattivo(fig)


if __name__ == "__main__":
    principale()
