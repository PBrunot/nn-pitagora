from manim import *
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense
from nn_pitagora import load_saved_model, crea_modello_epoca_zero


class TriangoliRettangoli(Scene):
    """
    Mostra più triangoli rettangoli con lati etichettati che mostrano esempi del teorema di Pitagora.
    Ogni triangolo mostra l'indicatore dell'angolo retto e le etichette delle lunghezze dei lati.
    """

    def construct(self):
        # Crea modello epoca 0 con pesi casuali se non esiste
        try:
            load_saved_model(epoch_number=0)
        except:
            print("Creando modello epoca 0 con pesi casuali...")
            crea_modello_epoca_zero()
            
        # Definisce set di terne pitagoriche (a, b, c)
        dati_triangoli = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)]

        # Crea un VGroup per contenere tutti i triangoli
        tutti_triangoli = VGroup()

        # Crea triangoli e lascia che Manim li organizzi nella zona sinistra
        for i, (a, b, c) in enumerate(dati_triangoli):
            gruppo_triangolo = self.crea_triangolo_rettangolo(a, b, c)
            tutti_triangoli.add(gruppo_triangolo)

        # Organizza triangoli nella zona sinistra automaticamente
        tutti_triangoli.arrange_in_grid(rows=2, cols=2, buff=0.5)
        tutti_triangoli.move_to(LEFT * 4.5)

        # Aggiungi etichette sezioni
        etichetta_input = Text("Triangoli Input", font_size=24)
        etichetta_input.move_to(LEFT * 4.5 + UP * 3.5)

        etichetta_nn = Text("Rete Neurale", font_size=24)
        etichetta_nn.move_to(UP * 3.5)

        etichetta_output = Text("Risultati", font_size=24)
        etichetta_output.move_to(RIGHT * 4.5 + UP * 3.5)

        # Anima tutto
        self.play(Write(etichetta_input), Write(etichetta_nn), Write(etichetta_output))
        self.wait(0.5)

        # Carica un modello per ottenere l'architettura
        try:
            modello_esempio = load_saved_model(epoch_number=0)
        except:
            modello_esempio = None

        # Crea e mostra la rete neurale
        rete_neurale = self.crea_rete_neurale(modello=modello_esempio)
        self.play(Create(rete_neurale), run_time=2)
        self.wait(1)

        # Anima triangoli input
        for triangolo in tutti_triangoli:
            self.play(Create(triangolo), run_time=1.5)
            self.wait(0.5)

        # Crea indicatore epoca persistente
        testo_epoca = Text("Epoca 0", font_size=28, color=YELLOW)
        testo_epoca.next_to(rete_neurale, DOWN, buff=0.5)
        self.play(Write(testo_epoca), run_time=0.5)
        self.wait(0.5)

        for epoch in [0, 1, 10, 75]:
            # Aggiorna indicatore epoca
            if epoch != 0:  # Non aggiornare per la prima epoca (già mostrata)
                nuovo_testo_epoca = Text(f"Epoca {epoch}", font_size=28, color=YELLOW)
                nuovo_testo_epoca.next_to(rete_neurale, DOWN, buff=0.5)
                self.play(Transform(testo_epoca, nuovo_testo_epoca), run_time=0.5)
                self.wait(0.5)

            # Carica predizioni NN epoca 1
            ipotenuse_predette = self.ottieni_predizioni_nn(dati_triangoli, epoch)

            # Anima flusso dati attraverso NN per il primo triangolo
            primo_triangolo = dati_triangoli[0]
            a_primo, b_primo, _ = primo_triangolo
            pred_c_primo = ipotenuse_predette[0]

            # Carica modello per questa epoca
            try:
                modello_epoca = load_saved_model(epoch_number=epoch)
            except:
                modello_epoca = None

            etichette_valori_nn = self.anima_flusso_dati_nn(
                rete_neurale,
                tutti_triangoli[0],
                a_primo,
                b_primo,
                pred_c_primo,
                modello_epoca,
            )
            self.wait(1)

            # Crea triangoli di output con predizioni NN
            tutti_triangoli_output = VGroup()
            for i, ((a, b, c), pred_c) in enumerate(
                zip(dati_triangoli, ipotenuse_predette)
            ):
                gruppo_triangolo_output = self.crea_triangolo_rettangolo(
                    a, b, pred_c, e_predizione=True
                )
                tutti_triangoli_output.add(gruppo_triangolo_output)

            # Organizza triangoli output nella zona destra
            tutti_triangoli_output.arrange_in_grid(rows=2, cols=2, buff=0.5)
            tutti_triangoli_output.move_to(RIGHT * 4.5)

            # Anima triangoli output
            for triangolo in tutti_triangoli_output:
                self.play(Create(triangolo), run_time=1.5)
                self.wait(0.5)

            # Rimuovi etichette valori dalla rete neurale
            self.play(FadeOut(etichette_valori_nn), run_time=0.5)

            self.wait(1)
            # Nascondi etichette triangoli input prima della sovrapposizione
            animazioni_dissolvenza = []
            for triangolo_input in tutti_triangoli:
                # Nascondi etichette (elementi 2, 3, 4 sono etichetta_a, etichetta_b, etichetta_c)
                # Struttura: [triangolo, angolo_retto, etichetta_a, etichetta_b, etichetta_c]
                if len(triangolo_input) >= 5:
                    for etichetta in triangolo_input[
                        2:
                    ]:  # Salta triangolo e angolo_retto
                        animazioni_dissolvenza.append(FadeOut(etichetta))

            self.play(*animazioni_dissolvenza, run_time=0.5)

            # Muovi triangoli output per sovrapporli ai triangoli input per mostrare differenze
            # Allinea ai vertici degli angoli retti (punto A = ORIGIN per ogni triangolo)
            animazioni_sovrapposizione = []
            for triangolo_input, triangolo_output in zip(
                tutti_triangoli, tutti_triangoli_output
            ):
                # Ottieni la posizione del poligono triangolo (primo elemento) in ogni gruppo
                poligono_triangolo_input = triangolo_input[
                    0
                ]  # Triangolo è primo elemento
                poligono_triangolo_output = triangolo_output[
                    0
                ]  # Triangolo è primo elemento

                # Calcola dove si trova il vertice dell'angolo retto per ogni triangolo
                vertice_retto_input = poligono_triangolo_input.get_vertices()[
                    0
                ]  # Punto A (angolo retto)
                vertice_retto_output = poligono_triangolo_output.get_vertices()[
                    0
                ]  # Punto A (angolo retto)

                # Calcola l'offset necessario per allineare i vertici degli angoli retti
                offset = vertice_retto_input - vertice_retto_output
                animazioni_sovrapposizione.append(
                    triangolo_output.animate.shift(offset)
                )

            self.play(*animazioni_sovrapposizione, run_time=2)
            self.wait(1)

            # Mostra analisi errore invece dell'ipotenusa predetta
            testo_errore_medio = self.mostra_analisi_errore(
                tutti_triangoli_output, dati_triangoli, ipotenuse_predette
            )
            self.wait(3)

            # Nascondi errore medio e lunghezze dei triangoli output
            animazioni_pulizia = []

            # Nascondi triangoli output con le loro etichette errore
            animazioni_pulizia.append(FadeOut(tutti_triangoli_output))

            # Nascondi il testo errore medio
            if testo_errore_medio is not None:
                animazioni_pulizia.append(FadeOut(testo_errore_medio))

            # Mantieni il testo epoca visibile per la prossima iterazione

            self.play(*animazioni_pulizia, run_time=0.5)
            self.wait(0.5)

            # Ricominciamo dai triangoli input per la prossima epoca
            animazioni_ricomparsa = []
            for triangolo_input in tutti_triangoli:
                # Fai riapparire etichette (elementi 2, 3, 4 sono etichetta_a, etichetta_b, etichetta_c)
                if len(triangolo_input) >= 5:
                    for etichetta in triangolo_input[
                        2:
                    ]:  # Salta triangolo e angolo_retto
                        animazioni_ricomparsa.append(FadeIn(etichetta))

            self.play(*animazioni_ricomparsa, run_time=0.5)
            self.wait(1)
        
        # Fase finale: mostra matrici di pesi e bias del modello finale addestrato
        try:
            modello_finale = load_saved_model(epoch_number=75)
            self.mostra_matrici_pesi_finali(modello_finale)
        except Exception as e:
            print(f"Errore nel caricamento del modello finale per mostrare i pesi: {e}")
            # Mostra messaggio finale senza matrici
            messaggio_finale = Text(
                "Addestramento Completato!\nIl modello ha imparato il Teorema di Pitagora",
                font_size=32,
                color=GREEN
            )
            messaggio_finale.move_to(ORIGIN)
            self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)
            self.play(Write(messaggio_finale), run_time=2)
            self.wait(3)

    def crea_triangolo_rettangolo(self, a, b, c, e_predizione=False):
        """
        Crea un triangolo rettangolo con lati etichettati e indicatore angolo retto.

        Args:
            a, b: Lunghezze dei lati (cateti input)
            c: Lunghezza ipotenusa (reale o predetta)
            e_predizione: Se questa è una predizione NN (influisce sulla colorazione)

        Returns:
            VGroup contenente il triangolo e tutte le etichette
        """
        # Fattore di scala per rendere i triangoli visibili ma non troppo grandi
        scala = 0.15

        # Definisce vertici del triangolo
        # Posiziona angolo retto all'origine per semplicità
        A = ORIGIN
        B = RIGHT * a * scala

        if e_predizione:
            # Per predizioni, mantieni a e b fissi, ma posiziona C per far corrispondere ipotenusa predetta
            # Vogliamo |BC| = c_predetta, con B a (a*scala, 0) e C da qualche parte
            # Posizioniamo C sul cerchio centrato in B con raggio c_predetta*scala
            # Per visualizzazione, posizioniamo C approssimativamente dove "dovrebbe" essere ma aggiustato
            # Inizia con la posizione corretta
            C_corretto = UP * b * scala
            # Calcola la distanza da B a C corretto
            distanza_corretta = np.linalg.norm(C_corretto - B)
            # Scala posizione C per corrispondere alla distanza predetta
            if distanza_corretta > 0:
                direzione = (C_corretto - B) / distanza_corretta
                C = B + direzione * c * scala
            else:
                C = UP * b * scala  # Fallback
        else:
            C = UP * b * scala

        # Crea triangolo con colore basato sullo stato di predizione
        colore_triangolo = ORANGE if e_predizione else BLUE
        triangolo = Polygon(A, B, C, color=colore_triangolo, fill_opacity=0.2)

        # Crea indicatore angolo retto (solo per triangoli input)
        componenti = [triangolo]
        if not e_predizione:
            angolo_retto = self.crea_indicatore_angolo_retto(A, B, C, dimensione=0.15)
            componenti.append(angolo_retto)

        # Crea etichette lati
        # Etichetta per lato 'a' (orizzontale)
        etichetta_a = Text(str(a), font_size=24, color=RED)
        etichetta_a.next_to((A + B) / 2, DOWN, buff=0.1)

        # Etichetta per lato 'b' (verticale) - mostra sempre valore b originale
        etichetta_b = Text(str(b), font_size=24, color=RED)
        etichetta_b.next_to((A + C) / 2, LEFT, buff=0.1)

        # Etichetta per ipotenusa 'c' (formato a 1 decimale per predizioni)
        testo_c = f"{c:.2f}" if e_predizione else str(c)
        colore_etichetta_c = YELLOW if e_predizione else GREEN
        etichetta_c = Text(testo_c, font_size=24, color=colore_etichetta_c)
        etichetta_c.next_to((B + C) / 2, UR, buff=0.1)

        # Raggruppa tutto insieme
        componenti.extend([etichetta_a, etichetta_b, etichetta_c])
        gruppo_triangolo = VGroup(*componenti)

        return gruppo_triangolo

    def crea_rete_neurale(self, modello=None):
        """
        Crea visualizzazione della rete neurale feed-forward.

        Args:
            modello: Modello Keras caricato (se None, usa configurazione default)

        Returns:
            VGroup contenente la visualizzazione della rete neurale
        """
        # Estrai numero di unità nascoste dal modello
        if modello is not None:
            # Il primo layer (layer[0]) è lo strato nascosto
            unita_nascoste = modello.layers[0].units
        else:
            unita_nascoste = 50  # Default

        # Configurazione visualizzazione
        raggio_neurone = 0.08
        spaziatura_verticale = 3 * raggio_neurone
        spaziatura_layer = 1.5

        # Limite di neuroni nascosti da visualizzare (per evitare sovraccarico visivo)
        max_neuroni_visualizzati = 20
        neuroni_da_visualizzare = min(unita_nascoste, max_neuroni_visualizzati)
        mostra_punti_sospensione = unita_nascoste > max_neuroni_visualizzati

        # Posizione centrale
        centro = ORIGIN

        # Layer di input (2 neuroni)
        input_neurons = VGroup()
        for i in range(2):
            neurone = Circle(radius=raggio_neurone, color=BLUE, fill_opacity=0.3)

            if i == 0:
                y_pos = -1
            else:
                y_pos = 1

            neurone.move_to(centro + LEFT * spaziatura_layer + DOWN * y_pos)
            input_neurons.add(neurone)

        # Etichette input
        etichetta_a = Text("a", font_size=20, color=BLUE)
        etichetta_a.next_to(input_neurons[0], LEFT, buff=0.2)
        etichetta_b = Text("b", font_size=20, color=BLUE)
        etichetta_b.next_to(input_neurons[1], LEFT, buff=0.2)

        # Layer nascosto
        hidden_neurons = VGroup()
        hidden_y_start = (neuroni_da_visualizzare - 1) * spaziatura_verticale / 2
        for i in range(neuroni_da_visualizzare):
            neurone = Circle(radius=raggio_neurone, color=YELLOW, fill_opacity=0.3)
            y_pos = hidden_y_start - i * spaziatura_verticale
            neurone.move_to(centro + UP * y_pos)
            hidden_neurons.add(neurone)

        # Punti di sospensione se ci sono più neuroni nascosti di quelli visualizzati
        punti_sospensione = VGroup()
        if mostra_punti_sospensione:
            punti = Text("...", font_size=24, color=YELLOW)
            punti.next_to(hidden_neurons[-1], DOWN, buff=0.1)
            punti_sospensione.add(punti)

        # Layer di output (1 neurone)
        output_neuron = Circle(radius=raggio_neurone, color=ORANGE, fill_opacity=0.3)
        output_neuron.move_to(centro + RIGHT * spaziatura_layer)

        # Etichetta output
        etichetta_c = Text("c", font_size=20, color=YELLOW)
        etichetta_c.next_to(output_neuron, RIGHT, buff=0.2)

        # Crea frecce (arrows) tra layer
        arrows = VGroup()

        # Input -> Hidden
        for input_n in input_neurons:
            for hidden_n in hidden_neurons:
                arrow = Arrow(
                    input_n.get_right(),
                    hidden_n.get_left(),
                    buff=0,
                    stroke_width=0.5,
                    max_tip_length_to_length_ratio=0.01,
                    color=GRAY,
                    fill_opacity=0.5,
                    tip_shape=StealthTip,
                )
                arrows.add(arrow)

        # Hidden -> Output
        for hidden_n in hidden_neurons:
            arrow = Arrow(
                hidden_n.get_right(),
                output_neuron.get_left(),
                buff=0,
                stroke_width=0.5,
                max_tip_length_to_length_ratio=0.01,
                color=GRAY,
                fill_opacity=0.5,
                tip_shape=StealthTip,
            )
            arrows.add(arrow)

        # Raggruppa tutto
        rete_neurale = VGroup(
            input_neurons,
            etichetta_a,
            etichetta_b,
            hidden_neurons,
            output_neuron,
            etichetta_c,
            arrows,
            punti_sospensione,
        )

        return rete_neurale

    def crea_indicatore_angolo_retto(self, vertice, punto1, punto2, dimensione=0.15):
        """
        Crea un indicatore angolo retto (piccolo quadrato) al vertice dato.

        Args:
            vertice: Il vertice dove si trova l'angolo retto
            punto1, punto2: Gli altri due punti del triangolo
            dimensione: Dimensione dell'indicatore angolo retto

        Returns:
            VGroup contenente l'indicatore angolo retto
        """
        # Calcola vettori unitari lungo i due lati
        vet1 = punto1 - vertice
        vet2 = punto2 - vertice

        # Normalizza vettori
        if np.linalg.norm(vet1) > 0:
            vet1 = vet1 / np.linalg.norm(vet1) * dimensione
        if np.linalg.norm(vet2) > 0:
            vet2 = vet2 / np.linalg.norm(vet2) * dimensione

        # Crea quadrato per indicatore angolo retto
        angolo1 = vertice
        angolo2 = vertice + vet1
        angolo3 = vertice + vet1 + vet2
        angolo4 = vertice + vet2

        quadrato_angolo_retto = Polygon(
            angolo1,
            angolo2,
            angolo3,
            angolo4,
            color=WHITE,
            fill_opacity=0,
            stroke_width=2,
        )

        return quadrato_angolo_retto

    def ottieni_predizioni_nn(self, dati_triangoli, epoch=1):
        """
        Ottieni predizioni NN per ipotenusa dal modello di una data epoca.

        Args:
            dati_triangoli: Lista di tuple (a, b, c)
            epoch: Numero dell'epoca del modello da usare

        Returns:
            Lista di valori ipotenusa predetti da NN per l'epoca specificata
        """
        try:
            # Carica il modello salvato per l'epoca specificata
            modello_epoca = load_saved_model(epoch_number=epoch)

            # Crea input test per i nostri triangoli
            input_test = np.array([[a, b] for a, b, c in dati_triangoli])

            # Ottieni predizioni reali dal modello
            predizioni = modello_epoca.predict(input_test, verbose=0)

            # Converte da array 2D a lista 1D
            predizioni_lista = [float(pred[0]) for pred in predizioni]

            print(f"Predizioni NN epoca {epoch}: {predizioni_lista}")
            return predizioni_lista

        except FileNotFoundError as e:
            print(
                f"Avvertimento: Modello epoca {epoch} non trovato ({e}), uso approssimazione"
            )
            # Fallback: ritorna approssimazioni grossolane per dimostrazione
            return [abs(a + b - c / 2) for a, b, c in dati_triangoli]
        except Exception as e:
            print(f"Errore nel caricamento modello epoca {epoch}: {e}")
            # Fallback: simula predizioni scarse (casuale ma deterministica)
            predizioni = []
            for a, b, c in dati_triangoli:
                predizione_scadente = (a + b) * 0.7 + np.random.RandomState(
                    42
                ).random() * 2
                predizioni.append(predizione_scadente)
            return predizioni

    def anima_flusso_dati_nn(
        self, rete_neurale, triangolo_input, a_val, b_val, pred_c, modello
    ):
        """
        Anima il flusso di dati attraverso la rete neurale.

        Args:
            rete_neurale: VGroup della rete neurale visualizzata
            triangolo_input: Il triangolo di input da cui prendere i valori
            a_val: Valore di a (cateto)
            b_val: Valore di b (cateto)
            pred_c: Valore predetto di c (ipotenusa)
            modello: Modello Keras per estrarre i pesi

        Returns:
            VGroup contenente le etichette dei valori sulla rete neurale (da rimuovere dopo)
        """
        # Estrai componenti della rete neurale
        arrows = rete_neurale[6]
        input_neurons = rete_neurale[0]
        hidden_neurons = rete_neurale[3]
        output_neuron = rete_neurale[4]

        # Ottieni le etichette a e b dal triangolo input
        # Struttura triangolo: [triangolo, angolo_retto, etichetta_a, etichetta_b, etichetta_c]
        etichetta_a_triangolo = triangolo_input[2]
        etichetta_b_triangolo = triangolo_input[3]

        # Crea copie delle etichette per l'animazione
        etichetta_a_mobile = etichetta_a_triangolo.copy()
        etichetta_b_mobile = etichetta_b_triangolo.copy()

        # Crea etichette dei valori per i neuroni input
        valore_a_neurone = Text(str(a_val), font_size=20, color=BLUE)
        valore_a_neurone.next_to(input_neurons[0], UP, buff=0.15)

        valore_b_neurone = Text(str(b_val), font_size=20, color=BLUE)
        valore_b_neurone.next_to(input_neurons[1], UP, buff=0.15)

        # Aggiungi le etichette mobili alla scena
        self.add(etichetta_a_mobile, etichetta_b_mobile)

        # Anima movimento di a e b verso i neuroni input
        self.play(
            Transform(etichetta_a_mobile, valore_a_neurone),
            Transform(etichetta_b_mobile, valore_b_neurone),
            run_time=0.5,
        )
        self.wait(0.5)

        # INTERMEDIATE STEP: Anima le frecce basandosi sui pesi del modello
        if modello is not None:
            # Estrai pesi e bias dal modello
            weights = modello.get_weights()
            W1 = weights[0]  # Pesi input -> hidden: shape (2, n_hidden)
            b1 = weights[1]  # Bias hidden layer: shape (n_hidden,)
            W2 = weights[2]  # Pesi hidden -> output: shape (n_hidden, 1)
            b2 = weights[3]  # Bias output layer: shape (1,)

            # Trova il peso massimo assoluto per normalizzazione larghezza
            max_peso_assoluto = max(np.abs(W1).max(), np.abs(W2).max())
            max_stroke = 5.0

            # Trova bias massimo assoluto per normalizzazione colore
            max_bias_assoluto = max(np.abs(b1).max(), abs(b2[0]) if len(b2) > 0 else 0)

            # Numero di neuroni nascosti visualizzati
            n_neuroni_visualizzati = len(hidden_neurons)
            n_neuroni_totali = W1.shape[1]

            # Calcola indici dei neuroni visualizzati
            indici_visualizzati = np.linspace(
                0, n_neuroni_totali - 1, n_neuroni_visualizzati, dtype=int
            )

            # Definisce colori per bias: DARK_GRAY per bias negativi, WHITE per bias positivi
            def ottieni_colore_da_bias(bias_val, max_bias):
                """Calcola colore basato sul valore del bias."""
                if max_bias == 0:
                    return GRAY
                # Normalizza bias tra -1 e 1
                bias_normalizzato = np.clip(bias_val / max_bias, -1, 1)
                # Interpola tra DARK_GRAY (-1) e WHITE (+1) passando per GRAY (0)
                if bias_normalizzato < 0:
                    # Bias negativo: interpola da DARK_GRAY a GRAY
                    return interpolate_color(DARK_GRAY, GRAY, (bias_normalizzato + 1))
                else:
                    # Bias positivo: interpola da GRAY a WHITE
                    return interpolate_color(GRAY, WHITE, bias_normalizzato)

            # Crea nuove frecce con larghezza basata sui pesi e colore basato sul bias
            animazioni_frecce = []

            # Input -> Hidden arrows (usano bias del layer hidden)
            arrow_idx = 0
            for i in range(2):  # 2 input neurons
                for j, h_idx in enumerate(indici_visualizzati):
                    peso = W1[i, h_idx]
                    bias = b1[h_idx]

                    peso_normalizzato = (
                        (abs(peso) / max_peso_assoluto) * max_stroke
                        if max_peso_assoluto > 0
                        else 0.5
                    )
                    colore = ottieni_colore_da_bias(bias, max_bias_assoluto)

                    # Ottieni la freccia corrente
                    freccia_corrente = arrows[arrow_idx]

                    # Crea nuova freccia con larghezza e colore aggiornati
                    nuova_freccia = Arrow(
                        input_neurons[i].get_right(),
                        hidden_neurons[j].get_left(),
                        buff=0,
                        stroke_width=peso_normalizzato,
                        max_tip_length_to_length_ratio=0.01,
                        color=colore,
                        fill_opacity=0.8,
                        tip_shape=StealthTip,
                    )

                    animazioni_frecce.append(Transform(freccia_corrente, nuova_freccia))
                    arrow_idx += 1

            # Hidden -> Output arrows (usano bias del layer output)
            bias_output = b2[0] if len(b2) > 0 else 0
            colore_output = ottieni_colore_da_bias(bias_output, max_bias_assoluto)

            for j, h_idx in enumerate(indici_visualizzati):
                peso = W2[h_idx, 0]
                peso_normalizzato = (
                    (abs(peso) / max_peso_assoluto) * max_stroke
                    if max_peso_assoluto > 0
                    else 0.5
                )

                freccia_corrente = arrows[arrow_idx]

                nuova_freccia = Arrow(
                    hidden_neurons[j].get_right(),
                    output_neuron.get_left(),
                    buff=0,
                    stroke_width=peso_normalizzato,
                    max_tip_length_to_length_ratio=0.01,
                    color=colore_output,
                    fill_opacity=0.8,
                    tip_shape=StealthTip,
                )

                animazioni_frecce.append(Transform(freccia_corrente, nuova_freccia))
                arrow_idx += 1

            # Anima tutte le frecce contemporaneamente
            self.play(*animazioni_frecce, run_time=1.5)
            self.wait(1)

        # Crea etichetta per il valore predetto al neurone output
        valore_pred_neurone = Text(f"{pred_c:.2f}", font_size=20, color=YELLOW)
        valore_pred_neurone.next_to(output_neuron, UP, buff=0.15)

        # Anima l'apparizione della predizione
        self.play(Write(valore_pred_neurone), run_time=1)
        self.wait(0.5)

        # Raggruppa tutte le etichette valori per poterle rimuovere dopo
        etichette_valori = VGroup(
            etichetta_a_mobile, etichetta_b_mobile, valore_pred_neurone
        )

        return etichette_valori

    def mostra_analisi_errore(
        self, triangoli_output, dati_triangoli, ipotenuse_predette
    ):
        """
        Mostra analisi errore sostituendo etichette ipotenusa predetta con valori errore.

        Args:
            triangoli_output: VGroup di gruppi triangoli output
            dati_triangoli: Dati triangoli originali [(a, b, c), ...]
            ipotenuse_predette: Lista di valori c predetti
        """
        # Calcola errori e percentuali
        errori = []
        percentuali_errore = []

        for (a, b, c_reale), pred_c in zip(dati_triangoli, ipotenuse_predette):
            errore = abs(c_reale - pred_c)
            errore_pct = (errore / c_reale) * 100 if c_reale > 0 else 0
            errori.append(errore)
            percentuali_errore.append(errore_pct)

        # Calcola errore medio
        errore_medio_pct = (
            sum(percentuali_errore) / len(percentuali_errore)
            if percentuali_errore
            else 0
        )

        # Sostituisci etichette ipotenusa con etichette errore
        animazioni_trasformazione = []

        for i, (triangolo_output, errore, errore_pct) in enumerate(
            zip(triangoli_output, errori, percentuali_errore)
        ):
            # Trova etichetta ipotenusa (ultimo elemento nel gruppo)
            etichetta_ipo = triangolo_output[
                -1
            ]  # Ultimo elemento dovrebbe essere etichetta_c

            # Crea nuova etichetta errore
            testo_errore = f"Δ{errore:.1f}\n({errore_pct:.1f}%)"
            nuova_etichetta_errore = Text(testo_errore, font_size=20, color=RED)
            nuova_etichetta_errore.move_to(etichetta_ipo.get_center())

            # Trasforma vecchia etichetta in nuova etichetta errore
            animazioni_trasformazione.append(
                Transform(etichetta_ipo, nuova_etichetta_errore)
            )

        # Mostra trasformazioni
        self.play(*animazioni_trasformazione, run_time=1.5)

        # Aggiungi visualizzazione errore medio sulla sinistra
        testo_errore_medio = Text(
            f"Errore Medio: {errore_medio_pct:.1f}%", font_size=32, color=RED
        )
        testo_errore_medio.to_corner(DL, buff=0.5)

        self.play(Write(testo_errore_medio), run_time=1)
        self.wait(2)

        # Ritorna il testo errore medio per poterlo rimuovere dopo
        return testo_errore_medio

    def mostra_matrici_pesi_finali(self, modello_finale):
        """
        Mostra le matrici di pesi e bias finali del modello addestrato in formato semplice e leggibile.

        Args:
            modello_finale: Modello Keras addestrato della epoch finale
        """
        if modello_finale is None:
            print("Nessun modello finale disponibile per mostrare i pesi")
            return

        # Estrai pesi e bias dal modello
        weights = modello_finale.get_weights()
        W1 = weights[0]  # Pesi input -> hidden: shape (2, n_hidden)
        b1 = weights[1]  # Bias hidden layer: shape (n_hidden,)
        W2 = weights[2]  # Pesi hidden -> output: shape (n_hidden, 1)
        b2 = weights[3]  # Bias output layer: shape (1,)

        n_hidden = W1.shape[1]

        # Nascondi tutti gli elementi correnti
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)

        # Titolo principale
        titolo = Text("Pesi e Bias del Modello Finale", font_size=36, color=YELLOW)
        titolo.to_edge(UP)
        self.play(Write(titolo), run_time=1)
        self.wait(0.5)

        # SCHERMATA 1: Matrice W1 e vettore b1 (strato nascosto)
        titolo_hidden = Text("Strato Nascosto (Input → Hidden)", font_size=28, color=GREEN)
        titolo_hidden.move_to(UP * 2.5)
        self.play(Write(titolo_hidden), run_time=0.5)

        # Crea rappresentazione matrice W1
        titolo_w1 = Text(f"Matrice W1 (2 × {n_hidden})", font_size=24)
        titolo_w1.move_to(UP * 1.8 + LEFT * 3)

        # Costruisci la matrice W1 come testo matematico (mostra prime e ultime colonne)
        max_show_cols = 6
        if n_hidden <= max_show_cols:
            w1_rows = []
            for i in range(2):
                row_vals = [f"{W1[i,j]:.2f}" for j in range(n_hidden)]
                w1_rows.append(" & ".join(row_vals))
            w1_matrix_str = r"\begin{bmatrix}" + r" \\ ".join(w1_rows) + r"\end{bmatrix}"
        else:
            show_first = 3
            show_last = 2
            w1_rows = []
            for i in range(2):
                row_vals = [f"{W1[i,j]:.2f}" for j in range(show_first)]
                row_vals.append(r"\cdots")
                row_vals.extend([f"{W1[i,j]:.2f}" for j in range(n_hidden-show_last, n_hidden)])
                w1_rows.append(" & ".join(row_vals))
            w1_matrix_str = r"\begin{bmatrix}" + r" \\ ".join(w1_rows) + r"\end{bmatrix}"

        w1_matrix = MathTex(w1_matrix_str, font_size=24)
        w1_matrix.next_to(titolo_w1, DOWN, buff=0.3)
        w1_matrix.shift(LEFT * 3)

        # Crea rappresentazione vettore b1
        titolo_b1 = Text(f"Vettore b1 ({n_hidden})", font_size=24)
        titolo_b1.move_to(UP * 1.8 + RIGHT * 3)

        if n_hidden <= max_show_cols:
            b1_vals = [f"{b1[j]:.2f}" for j in range(n_hidden)]
            b1_vector_str = r"\begin{bmatrix}" + r" \\ ".join(b1_vals) + r"\end{bmatrix}"
        else:
            show_first = 3
            show_last = 2
            b1_vals = [f"{b1[j]:.2f}" for j in range(show_first)]
            b1_vals.append(r"\vdots")
            b1_vals.extend([f"{b1[j]:.2f}" for j in range(n_hidden-show_last, n_hidden)])
            b1_vector_str = r"\begin{bmatrix}" + r" \\ ".join(b1_vals) + r"\end{bmatrix}"

        b1_vector = MathTex(b1_vector_str, font_size=24)
        b1_vector.next_to(titolo_b1, DOWN, buff=0.3)
        b1_vector.shift(RIGHT * 3)

        self.play(Write(titolo_w1), Write(titolo_b1))
        self.play(FadeIn(w1_matrix), FadeIn(b1_vector))
        self.wait(3)

        # SCHERMATA 2: Matrice W2 e scalare b2 (strato output)
        self.play(
            FadeOut(titolo_hidden), FadeOut(titolo_w1), FadeOut(titolo_b1),
            FadeOut(w1_matrix), FadeOut(b1_vector)
        )

        titolo_output = Text("Strato Output (Hidden → Output)", font_size=28, color=RED)
        titolo_output.move_to(UP * 2.5)
        self.play(Write(titolo_output), run_time=0.5)

        # Crea rappresentazione matrice W2
        titolo_w2 = Text(f"Matrice W2 ({n_hidden} × 1)", font_size=24)
        titolo_w2.move_to(UP * 1.8 + LEFT * 3)

        if n_hidden <= max_show_cols:
            w2_vals = [f"{W2[j,0]:.2f}" for j in range(n_hidden)]
            w2_matrix_str = r"\begin{bmatrix}" + r" \\ ".join(w2_vals) + r"\end{bmatrix}"
        else:
            show_first = 3
            show_last = 2
            w2_vals = [f"{W2[j,0]:.2f}" for j in range(show_first)]
            w2_vals.append(r"\vdots")
            w2_vals.extend([f"{W2[j,0]:.2f}" for j in range(n_hidden-show_last, n_hidden)])
            w2_matrix_str = r"\begin{bmatrix}" + r" \\ ".join(w2_vals) + r"\end{bmatrix}"

        w2_matrix = MathTex(w2_matrix_str, font_size=24)
        w2_matrix.next_to(titolo_w2, DOWN, buff=0.3)
        w2_matrix.shift(LEFT * 3)

        # Crea rappresentazione scalare b2
        titolo_b2 = Text("Scalare b2", font_size=24)
        titolo_b2.move_to(UP * 1.8 + RIGHT * 3)

        b2_scalar = MathTex(f"{b2[0]:.4f}", font_size=32)
        b2_scalar.next_to(titolo_b2, DOWN, buff=0.3)
        b2_scalar.shift(RIGHT * 3)

        self.play(Write(titolo_w2), Write(titolo_b2))
        self.play(FadeIn(w2_matrix), FadeIn(b2_scalar))
        self.wait(3)

        # Messaggio finale
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)
