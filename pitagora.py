from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import plot_model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import random
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import tensorflow as tf

# Configura GPU se disponibile
print("\n" + "="*70)
print("CONFIGURAZIONE GPU")
print("="*70)

# Lista dispositivi disponibili
physical_devices = tf.config.list_physical_devices()
print(f"Dispositivi fisici disponibili: {len(physical_devices)}")
for device in physical_devices:
    print(f"  - {device.device_type}: {device.name}")

# Verifica GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Abilita memory growth per evitare di allocare tutta la memoria GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print(f"\n✓ GPU DISPONIBILE: {len(gpus)} dispositivo/i GPU trovato/i")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")

        # Imposta la GPU come dispositivo predefinito
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"  Dispositivi logici GPU: {len(logical_gpus)}")

    except RuntimeError as e:
        print(f"Errore nella configurazione GPU: {e}")
else:
    print("\n✗ Nessuna GPU trovata. Utilizzo CPU.")

print("="*70 + "\n")

random.seed(1809)
np.random.seed(1809)
tf.random.set_seed(1809)


def modello_pitagora(num_neuroni, learning_rate=0.001):
    model = Sequential(
        [
            Input(shape=(2,)),
            Dense(num_neuroni, activation="relu", name="Strato_interno"),
            Dense(1, activation="linear", name="Output_ipotenusa"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


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

def get_samples(n_samples = 10000):
    # Genera dati di training
    cateto1 = np.random.uniform(1, 50, n_samples)  # Cateto A: 1-10
    cateto2 = np.random.uniform(1, 50, n_samples)  # Cateto B: 1-10
    ipotenusa = np.sqrt(cateto1**2 + cateto2**2)  # Target esatto

    X_train = np.column_stack((cateto1, cateto2))  # Input (3000, 2)
    y_train = ipotenusa.reshape(-1, 1)  # Output (3000, 1)
    return X_train, y_train

X_train, y_train = get_samples(10000)

def Test_Pitagora(model):
    # Esempio: cateti 3 e 4 → ipotenusa ~5
    input_test = np.array([[3.0, 4.0], [6.0, 8.0]])
    risultati = model.predict(input_test, verbose=0)
    for i, risultato in enumerate(risultati):
        print(
            f"Ipotenusa di {input_test[i][0]} e {input_test[i][1]}: {risultato[0]:.4f}"
        )

    return risultati


def objective(trial):
    """Funzione obiettivo per Optuna: ottimizza iperparametri della rete neurale."""

    # Suggerisci iperparametri da testare
    num_neuroni = trial.suggest_int("num_neuroni", 5, 100, step=5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    num_samples = trial.suggest_int("num_samples", 500, 10000, log=True)
    # Crea e compila il modello
    model = modello_pitagora(num_neuroni, learning_rate)
    X_train, y_train = get_samples(num_samples)
    # Addestra il modello
    history = model.fit(
        X_train,
        y_train,
        epochs=100,  # Ridotto per velocizzare l'ottimizzazione
        batch_size=32,
        validation_split=0.2,
        verbose=0,  # Silenzioso per non intasare l'output
        shuffle=True,
    )

    return history.history["val_loss"][-1]


def objective_learning_rate(trial):
    """Fase 1: Ottimizza solo il learning rate con parametri fissi."""
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Parametri fissi per questa fase
    num_neuroni = 50
    num_samples = 5000

    model = modello_pitagora(num_neuroni, learning_rate)
    X_train, y_train = get_samples(num_samples)

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,  # Fisso a 16 (mini-batch gradient descent)
        validation_split=0.2,
        verbose=0,
        shuffle=True,
    )

    return history.history["val_loss"][-1]


def objective_with_best_lr(trial, best_learning_rate):
    """Fase 2: Ottimizza altri parametri con il miglior learning rate."""
    num_neuroni = trial.suggest_int("num_neuroni", 5, 100, step=5)
    num_samples = trial.suggest_int("num_samples", 500, 10000, log=True)

    # Usa il learning rate ottimizzato dalla fase 1
    model = modello_pitagora(num_neuroni, best_learning_rate)
    X_train, y_train = get_samples(num_samples)

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,  # Fisso a 16 (mini-batch gradient descent)
        validation_split=0.2,
        verbose=0,
        shuffle=True,
    )

    return history.history["val_loss"][-1]


def ottimizza_iperparametri(n_trials_lr=20, n_trials_other=30, sequential=True):
    """Esegue l'ottimizzazione degli iperparametri con Optuna.

    Args:
        n_trials_lr: Numero di trial per ottimizzare il learning rate (fase 1)
        n_trials_other: Numero di trial per ottimizzare altri parametri (fase 2)
        sequential: Se True, ottimizza prima LR poi altri. Se False, ottimizza tutto insieme.
    """

    if sequential:
        # FASE 1: Ottimizza Learning Rate
        print("\n" + "="*70)
        print("FASE 1: OTTIMIZZAZIONE LEARNING RATE")
        print("="*70)
        print(f"Numero di trial: {n_trials_lr}")
        print("Parametri fissi:")
        print("  - num_neuroni: 50")
        print("  - num_samples: 5000")
        print("  - batch_size: 16 (mini-batch)")
        print("\nParametro da ottimizzare:")
        print("  - learning_rate: [1e-5, 1e-2] (scala log)")
        print("="*70 + "\n")

        study_lr = optuna.create_study(
            direction="minimize",
            study_name="phase1_learning_rate",
            sampler=optuna.samplers.TPESampler(seed=1809)
        )

        study_lr.optimize(objective_learning_rate, n_trials=n_trials_lr, show_progress_bar=True)

        best_lr = study_lr.best_params["learning_rate"]

        print("\n" + "="*70)
        print("RISULTATI FASE 1")
        print("="*70)
        print(f"Miglior validation loss: {study_lr.best_value:.6f}")
        print(f"Miglior learning rate: {best_lr:.6f}")
        print("="*70 + "\n")

        # FASE 2: Ottimizza altri parametri con il miglior LR
        print("\n" + "="*70)
        print("FASE 2: OTTIMIZZAZIONE ALTRI PARAMETRI")
        print("="*70)
        print(f"Numero di trial: {n_trials_other}")
        print(f"Learning rate fisso (ottimizzato): {best_lr:.6f}")
        print("\nParametri da ottimizzare:")
        print("  - num_neuroni: [5, 10, 15, ..., 100]")
        print("  - num_samples: [500, 10000] (scala log)")
        print("\nParametro fisso:")
        print("  - batch_size: 16 (mini-batch)")
        print("="*70 + "\n")

        study_other = optuna.create_study(
            direction="minimize",
            study_name="phase2_other_params",
            sampler=optuna.samplers.TPESampler(seed=1809)
        )

        # Crea funzione obiettivo con LR fisso
        def objective_wrapper(trial):
            return objective_with_best_lr(trial, best_lr)

        study_other.optimize(objective_wrapper, n_trials=n_trials_other, show_progress_bar=True)

        print("\n" + "="*70)
        print("RISULTATI FASE 2")
        print("="*70)
        print(f"Miglior validation loss: {study_other.best_value:.6f}")
        print("\nMigliori iperparametri trovati:")
        print(f"  learning_rate: {best_lr:.6f} (da Fase 1)")
        for param, value in study_other.best_params.items():
            print(f"  {param}: {value}")
        print("="*70 + "\n")

        # Combina i risultati
        best_params = study_other.best_params.copy()
        best_params["learning_rate"] = best_lr

        # Salva visualizzazioni
        try:
            fig1_lr = plot_optimization_history(study_lr)
            fig1_lr.write_html("optuna_phase1_lr_history.html")
            print("Salvato: optuna_phase1_lr_history.html")

            fig2_other = plot_optimization_history(study_other)
            fig2_other.write_html("optuna_phase2_other_history.html")
            print("Salvato: optuna_phase2_other_history.html")

            fig3_importance = plot_param_importances(study_other)
            fig3_importance.write_html("optuna_param_importances.html")
            print("Salvato: optuna_param_importances.html")
        except Exception as e:
            print(f"Errore nel salvare le visualizzazioni: {e}")

        # Crea un oggetto simile a study per compatibilità
        class CombinedStudy:
            def __init__(self, best_params, best_value):
                self.best_params = best_params
                self.best_value = best_value

        return CombinedStudy(best_params, study_other.best_value)

    else:
        # Ottimizzazione simultanea (metodo originale)
        print("\n" + "="*70)
        print("OTTIMIZZAZIONE SIMULTANEA IPERPARAMETRI")
        print("="*70)
        print(f"Numero di trial: {n_trials_lr + n_trials_other}")
        print("Iperparametri da ottimizzare:")
        print("  - num_neuroni: [5, 10, 15, ..., 100]")
        print("  - learning_rate: [1e-5, 1e-2] (scala log)")
        print("  - num_samples: [500, 10000] (scala log)")
        print("="*70 + "\n")

        study = optuna.create_study(
            direction="minimize",
            study_name="pythagorean_nn_optimization",
            sampler=optuna.samplers.TPESampler(seed=1809)
        )

        study.optimize(objective, n_trials=n_trials_lr + n_trials_other, show_progress_bar=True)

        print("\n" + "="*70)
        print("RISULTATI OTTIMIZZAZIONE")
        print("="*70)
        print(f"Miglior validation loss: {study.best_value:.6f}")
        print("\nMigliori iperparametri trovati:")
        for param, value in study.best_params.items():
            print(f"  {param}: {value}")
        print("="*70 + "\n")

        try:
            fig1 = plot_optimization_history(study)
            fig1.write_html("optuna_optimization_history.html")
            print("Salvato: optuna_optimization_history.html")

            fig2 = plot_param_importances(study)
            fig2.write_html("optuna_param_importances.html")
            print("Salvato: optuna_param_importances.html")
        except Exception as e:
            print(f"Errore nel salvare le visualizzazioni: {e}")

        return study


def addestra_modello_finale(num_neuroni, learning_rate, epochs=300, num_samples=10000):
    """Addestra il modello finale con i migliori iperparametri."""
    print("\n" + "="*70)
    print("ADDESTRAMENTO MODELLO FINALE")
    print("="*70)
    print(f"num_neuroni: {num_neuroni}")
    print(f"learning_rate: {learning_rate}")
    print(f"batch_size: 16 (mini-batch)")
    print(f"num_samples: {num_samples}")
    print(f"epochs: {epochs}")
    print("="*70 + "\n")

    model = modello_pitagora(num_neuroni, learning_rate)

    print(model.summary())

    # Usa i campioni ottimizzati
    X_train_final, y_train_final = get_samples(num_samples)

    history = model.fit(
        X_train_final,
        y_train_final,
        epochs=epochs,
        batch_size=32,  # Fisso a 16 (mini-batch gradient descent)
        validation_split=0.2,
        verbose=1,
        shuffle=True,
    )

    # Test del modello
    print("\n" + "="*70)
    print("TEST DEL MODELLO")
    print("="*70)
    Test_Pitagora(model)
    print("="*70 + "\n")

    # Salva modello
    model.save("Rete_Pitagora.keras")
    print("Modello salvato in: Rete_Pitagora.keras\n")

    # Visualizza pesi finali
    print_model_weights(model)

    # Grafico loss
    fig = plt.figure(figsize=(15, 5))
    plt.plot(history.history["loss"], label="Training Loss", linewidth=2)
    plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    plt.title(f"Loss per Epoch (neuroni={num_neuroni}, lr={learning_rate:.6f})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log scale)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig("rete_ottimizzata.png", dpi=200, bbox_inches="tight")
    print("Grafico salvato in: rete_ottimizzata.png\n")
    plt.show()

    return model, history


def confronta_performance_campioni(sample_sizes=[10, 100, 1000, 10000, 100000],
                                   num_neuroni=50,
                                   learning_rate=0.001,
                                   epochs=50,
                                   test_samples=1000):
    """Confronta la performance della rete neurale con diversi numeri di campioni di training.

    Args:
        sample_sizes: Lista di numeri di campioni da testare
        num_neuroni: Numero di neuroni nello strato nascosto
        learning_rate: Learning rate per l'ottimizzatore
        epochs: Numero di epoche di training
        test_samples: Numero di campioni per il set di test (fisso per tutti)
    """

    print("\n" + "="*70)
    print("CONFRONTO PERFORMANCE CON DIVERSI NUMERI DI CAMPIONI")
    print("="*70)
    print(f"Configurazione:")
    print(f"  - Neuroni: {num_neuroni}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Epoche: {epochs}")
    print(f"  - Campioni da testare: {sample_sizes}")
    print(f"  - Campioni test (fisso): {test_samples}")
    print("="*70 + "\n")

    # Genera un set di test fisso per confronto equo
    print("Generazione set di test (fisso per tutti gli esperimenti)...")
    X_test, y_test = get_samples(test_samples)

    results = {
        'sample_sizes': [],
        'train_loss': [],
        'test_loss': [],
        'test_mae': []
    }

    for n_samples in sample_sizes:
        print(f"\n{'='*70}")
        print(f"Training con {n_samples} campioni...")
        print(f"{'='*70}")

        # Genera dati di training
        X_train, y_train = get_samples(n_samples)

        # Crea e addestra il modello
        model = modello_pitagora(num_neuroni, learning_rate)

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=min(32, n_samples),  # Adatta batch size per campioni piccoli
            verbose='auto',
            shuffle=True,
        )

        # Valuta sul set di test
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose='auto')
        train_loss = history.history["loss"][-1]

        # Salva risultati
        results['sample_sizes'].append(n_samples)
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        results['test_mae'].append(test_mae)

        print(f"  Training Loss finale: {train_loss:.6f}")
        print(f"  Test Loss: {test_loss:.6f}")
        print(f"  Test MAE: {test_mae:.6f}")

    print(f"\n{'='*70}")
    print("RISULTATI FINALI")
    print(f"{'='*70}")
    for i, n in enumerate(results['sample_sizes']):
        print(f"Campioni: {n:5d} | Train Loss: {results['train_loss'][i]:.6f} | "
              f"Test Loss: {results['test_loss'][i]:.6f} | Test MAE: {results['test_mae'][i]:.6f}")
    print(f"{'='*70}\n")

    # Crea grafici
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Grafico 1: Training Loss
    axes[0].plot(results['sample_sizes'], results['train_loss'],
                 marker='o', linewidth=2, markersize=8, color='blue')
    axes[0].set_xlabel('Numero di Campioni di Training (log scale)', fontsize=12)
    axes[0].set_ylabel('Training Loss (MSE, log scale)', fontsize=12)
    axes[0].set_title('Training Loss vs Numero di Campioni', fontsize=14, fontweight='bold')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3, which='both', linestyle='--')
    axes[0].set_xticks(results['sample_sizes'])
    axes[0].set_xticklabels(results['sample_sizes'])

    # Grafico 2: Test Loss
    axes[1].plot(results['sample_sizes'], results['test_loss'],
                 marker='s', linewidth=2, markersize=8, color='red')
    axes[1].set_xlabel('Numero di Campioni di Training (log scale)', fontsize=12)
    axes[1].set_ylabel('Test Loss (MSE, log scale)', fontsize=12)
    axes[1].set_title('Test Loss vs Numero di Campioni', fontsize=14, fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3, which='both', linestyle='--')
    axes[1].set_xticks(results['sample_sizes'])
    axes[1].set_xticklabels(results['sample_sizes'])

    # Grafico 3: Test MAE
    axes[2].plot(results['sample_sizes'], results['test_mae'],
                 marker='^', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Numero di Campioni di Training (log scale)', fontsize=12)
    axes[2].set_ylabel('Test MAE (log scale)', fontsize=12)
    axes[2].set_title('Test MAE vs Numero di Campioni', fontsize=14, fontweight='bold')
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3, which='both', linestyle='--')
    axes[2].set_xticks(results['sample_sizes'])
    axes[2].set_xticklabels(results['sample_sizes'])

    plt.tight_layout()
    plt.savefig("confronto_performance_campioni.png", dpi=200, bbox_inches="tight")
    print("Grafico salvato in: confronto_performance_campioni.png\n")
    plt.show()

    return results


if __name__ == "__main__":
    # Configurazione ottimizzazione
    USE_OPTUNA = False  # Imposta False per usare parametri predefiniti
    SEQUENTIAL_OPTIMIZATION = True  # True: ottimizza prima LR poi altri; False: tutto insieme
    CONFRONTA_CAMPIONI = True  # Imposta True per confrontare performance con diversi campioni

    if USE_OPTUNA:
        # Esegui ottimizzazione
        if SEQUENTIAL_OPTIMIZATION:
            # Ottimizzazione sequenziale: prima LR, poi altri parametri
            study = ottimizza_iperparametri(
                n_trials_lr=20,      # Trial per learning rate
                n_trials_other=30,   # Trial per altri parametri
                sequential=True
            )
        else:
            # Ottimizzazione simultanea di tutti i parametri
            study = ottimizza_iperparametri(
                n_trials_lr=25,
                n_trials_other=25,
                sequential=False
            )

        # Usa i migliori parametri trovati
        best_params = study.best_params
        model, history = addestra_modello_finale(
            num_neuroni=best_params["num_neuroni"],
            learning_rate=best_params["learning_rate"],
            num_samples=best_params.get("num_samples", 10000),
            epochs=100
        )
    else:
        # Modalità 2: Usa parametri predefiniti (veloce)
        print("Usando parametri predefiniti (salta ottimizzazione)...\n")

        if CONFRONTA_CAMPIONI:
            # Confronta performance con diversi numeri di campioni
            results = confronta_performance_campioni(
                sample_sizes=[10, 100, 1000, 10000, 100000],
                num_neuroni=50,
                learning_rate=0.001,
                epochs=50,
                test_samples=1000
            )
        else:
            # Training normale con un solo numero di campioni
            model, history = addestra_modello_finale(
                num_neuroni=50,
                learning_rate=0.001,
                num_samples=5000,
                epochs=50
            )
