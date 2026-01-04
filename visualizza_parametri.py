from manim import *
from keras.models import load_model
import numpy as np
import os
import math


class VisualizzaParametri(Scene):
    def get_color_from_value(self, value, min_val, max_val):
        """Mappa un valore numerico a un colore da blu (minore) a rosso (maggiore)"""
        if max_val == min_val:
            return BLUE
        # Normalizza il valore tra 0 e 1
        normalized = (value - min_val) / (max_val - min_val)
        # Interpola tra blu e rosso
        return interpolate_color(BLUE, RED, normalized)

    def construct(self):
        # Epoche da visualizzare
        epoche = [1, 5, 10, 20, 30, 40, 50, 100, 150, 200, 250]

        # Input di test: cateti 3 e 4 (ipotenusa attesa: 5)
        input_test = np.array([[3.0, 4.0]])

        # Genera dati di validazione (stesso seed del training)
        np.random.seed(1809)
        numero_campioni = 2000
        cateti1 = np.random.uniform(0, 15, numero_campioni)
        cateti2 = np.random.uniform(0, 15, numero_campioni)
        ipotenuse = np.sqrt(cateti1**2 + cateti2**2)
        cateti_val = np.column_stack((cateti1, cateti2))
        ipotenuse_val = ipotenuse.reshape(-1, 1)
        # Usa il 20% finale come validazione (come nel training)
        split_idx = int(numero_campioni * 0.8)
        X_val = cateti_val[split_idx:]
        y_val = ipotenuse_val[split_idx:]

        # Variabili per tracciare elementi persistenti
        tabella_precedente = None
        titolo_epoca = None
        esempio_label = None
        risultato_text = None

        # Variabili per il grafico MAE
        mae_values = []
        epoch_numbers = []
        grafico_assi = None
        x_label = None
        y_label = None
        punti_grafico = VGroup()
        linea_grafico = None
        epoca_label = None

        for idx, epoca in enumerate(epoche):
            # Carica il modello per questa epoca
            model_path = f"saved_models/model_epoch_{epoca:03d}.keras"

            if not os.path.exists(model_path):
                print(f"Modello non trovato: {model_path}")
                continue

            model = load_model(model_path)

            # Estrai i pesi
            layer0_weights = model.layers[0].get_weights()
            layer1_weights = model.layers[1].get_weights()

            W0 = layer0_weights[0]  # (2, 15) - pesi input -> hidden
            b0 = layer0_weights[1]  # (15,) - bias hidden layer
            W1 = layer1_weights[0]  # (15, 1) - pesi hidden -> output
            b1 = layer1_weights[1]  # (1,) - bias output layer

            # Raccogli tutti i valori numerici per determinare min/max
            tutti_valori = []
            tutti_valori.extend(W0.flatten())
            tutti_valori.extend(b0.flatten())
            tutti_valori.extend(W1.flatten())
            tutti_valori.extend(b1.flatten())
            min_val = min(tutti_valori)
            max_val = max(tutti_valori)

            # Crea la matrice dei parametri
            # Formato: [Neurone | weight_i1 | weight_i2 | bias_interno | weight_output | bias_output]
            matrice_dati = []

            # Header
            header = ["Nodo", "Peso i1", "Peso i2", "bias", "Peso out", "bias out"]
            matrice_dati.append(header)

            # 15 righe per i neuroni dello strato nascosto
            for i in range(15):
                riga = [
                    str(i+1),
                    f"{W0[0, i]:.3f}",
                    f"{W0[1, i]:.3f}",
                    f"{b0[i]:.3f}",
                    f"{W1[i, 0]:.3f}",
                    f"{b1[0]:.3f}" if i == 0 else ""
                ]
                matrice_dati.append(riga)

            # Crea la tabella Manim - POSIZIONATA A SINISTRA senza bordi
            tabella = Table(
                matrice_dati,
                include_outer_lines=False,  # Nessun bordo esterno
                line_config={"stroke_width": 0},  # Nessun bordo interno
                element_to_mobject_config={
                    "font_size": 20,
                    "font": "Monospace",  # Font monospace per allineamento costante
                },
                v_buff=0.1,  # Ridotto spazio verticale tra righe
                h_buff=0.3,  # Spazio orizzontale per compensare font monospace
                col_labels=None,
                row_labels=None,
            )

            # Imposta le celle alla stessa dimensione (dimensione fissa)
            # Non usiamo set_width/set_height per evitare problemi con le dimensioni dinamiche
            # La scala fissa della tabella garantisce già dimensioni costanti

            # Posiziona la tabella nella metà sinistra dello schermo
            tabella.move_to(LEFT * 3)

            # Colora l'header
            for j in range(6):
                tabella.get_entries((1, j+1)).set_color(YELLOW)

            # Colora i valori numerici in base al valore (blu -> rosso) e rendi grassetto
            for i in range(15):  # 15 righe di dati
                row_idx = i + 2  # +2 perché row 1 è l'header
                # Colonna 2: W_i1
                val = W0[0, i]
                cell = tabella.get_entries((row_idx, 2))
                cell.set_color(self.get_color_from_value(val, min_val, max_val))
                cell.set_weight(BOLD)
                # Colonna 3: W_i2
                val = W0[1, i]
                cell = tabella.get_entries((row_idx, 3))
                cell.set_color(self.get_color_from_value(val, min_val, max_val))
                cell.set_weight(BOLD)
                # Colonna 4: b_hid
                val = b0[i]
                cell = tabella.get_entries((row_idx, 4))
                cell.set_color(self.get_color_from_value(val, min_val, max_val))
                cell.set_weight(BOLD)
                # Colonna 5: W_out
                val = W1[i, 0]
                cell = tabella.get_entries((row_idx, 5))
                cell.set_color(self.get_color_from_value(val, min_val, max_val))
                cell.set_weight(BOLD)
                # Colonna 6: b_out (solo prima riga)
                if i == 0:
                    val = b1[0]
                    cell = tabella.get_entries((row_idx, 6))
                    cell.set_color(self.get_color_from_value(val, min_val, max_val))
                    cell.set_weight(BOLD)

            # Esegui la predizione
            predizione = model.predict(input_test, verbose=0)
            ipotenusa_predetta = predizione[0][0]

            # Calcola MAE di validazione
            y_pred = model.predict(X_val, verbose=0)
            val_mae = np.mean(np.abs(y_val - y_pred))
            mae_values.append(val_mae)
            epoch_numbers.append(epoca)

            # PANNELLO DESTRO - Informazioni
            # Titolo epoca
            nuovo_titolo = Text(f"Epoca {epoca}", font_size=48).shift(RIGHT * 3.5 + UP * 2.5)

            # Label esempio
            nuovo_esempio = Text("Esempio: cateti 3 e 4", font_size=32).shift(RIGHT * 3.5 + UP * 1)

            # Risultato
            nuovo_risultato = Text(
                f"Predizione: {ipotenusa_predetta:.4f}",
                font_size=32,
                color=GREEN
            ).shift(RIGHT * 3.5)

            # Gruppo pannello destro
            pannello_destro = VGroup(nuovo_titolo, nuovo_esempio, nuovo_risultato)

            # GRAFICO MAE - Parte inferiore destra (scala fissa 0-10)
            if idx == 0:
                # Crea gli assi del grafico con scala fissa 0-10 e numeri sugli assi
                grafico_assi = Axes(
                    x_range=[0, 250, 50],
                    y_range=[0, 10, 2],  # MAE da 0 a 10, step 2
                    x_length=3.5,
                    y_length=4,
                    axis_config={
                        "color": WHITE,
                        "include_numbers": True,
                        "font_size": 12,
                    },
                    tips=False,
                ).shift(RIGHT * 1.9 + DOWN * 1)

                # Label assi
                x_label = Text("Epoca", font_size=14).next_to(grafico_assi, DOWN, buff=0.6)
                y_label = Text("MAE", font_size=14).next_to(grafico_assi, LEFT, buff=0.2)

                # Primo punto
                punto = Dot(
                    grafico_assi.c2p(epoca, val_mae),
                    color=BLUE,
                    radius=0.04
                )
                punti_grafico.add(punto)

                # Legenda del grafico (sotto il grafico)
                epoca_label = Text("Progresso allenamento", font_size=14, color=BLUE).next_to(grafico_assi, DOWN, buff=0.2)

                # Prima epoca: mostra tutto
                self.play(Create(tabella), Write(pannello_destro))
                self.play(Create(grafico_assi), Write(x_label), Write(y_label))
                self.play(Create(punto), Write(epoca_label))
                self.wait(2)

                tabella_precedente = tabella
                titolo_epoca = nuovo_titolo
                esempio_label = nuovo_esempio
                risultato_text = nuovo_risultato
            else:
                # Epoche successive: aggiungi punto e linea
                if grafico_assi is not None:
                    nuovo_punto = Dot(
                        grafico_assi.c2p(epoca, val_mae),
                        color=BLUE,
                        radius=0.04
                    )

                    # Crea linea dal punto precedente al nuovo punto
                    ultimo_punto = punti_grafico[-1]
                    nuova_linea = Line(
                        ultimo_punto.get_center(),
                        nuovo_punto.get_center(),
                        color=BLUE,
                        stroke_width=2
                    )


                    self.play(
                        Transform(tabella_precedente, tabella),
                        Transform(titolo_epoca, nuovo_titolo),
                        Transform(risultato_text, nuovo_risultato),
                        Create(nuova_linea),
                        Create(nuovo_punto)
                    )
                    punti_grafico.add(nuovo_punto)

                    self.wait(1.5)

        # Finale
        self.wait(2)
