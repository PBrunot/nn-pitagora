from manim import *
import numpy as np
from keras.models import load_model
import pickle

class PesiReteNeurale(Scene):
    def construct(self):
        # Carica pesi modello addestrato
        try:
            with open('model_weights.pkl', 'rb') as f:
                dati_pesi = pickle.load(f)
            W1 = dati_pesi['W1']  # Forma: (2, 100)
            b1 = dati_pesi['b1']  # Forma: (100,)
            W2 = dati_pesi['W2']  # Forma: (100, 1)
            b2 = dati_pesi['b2']  # Forma: (1,)
        except FileNotFoundError:
            self.add(Text("Errore: model_weights.pkl non trovato!", color=RED))
            return

        # Titolo
        titolo = Text("Rete Neurale: f(a,b) = √(a² + b²)", font_size=36)
        titolo.to_edge(UP)
        self.play(Write(titolo))
        self.wait(0.5)

        # Ottieni dimensione rete reale
        n_neuroni_nascosti = W1.shape[1]

        # Testo architettura rete
        testo_arch = Text(f"Architettura: 2 → {n_neuroni_nascosti} (ReLU) → 1 (Lineare)", font_size=24)
        testo_arch.next_to(titolo, DOWN)
        self.play(FadeIn(testo_arch))
        self.wait(0.5)

        # Crea visualizzazione rete semplificata (mostra tutti se <=15, altrimenti mostra 15)
        n_neuroni_visualizzati = min(n_neuroni_nascosti, 15)

        # Strato input (2 neuroni)
        neuroni_input = VGroup()
        etichette_input = ["a", "b"]
        for i in range(2):
            neurone = Circle(radius=0.3, color=BLUE, fill_opacity=0.5)
            neurone.move_to(LEFT * 5 + UP * (1 - i) * 1.5)
            etichetta = Text(etichette_input[i], font_size=24)
            etichetta.move_to(neurone.get_center())
            neuroni_input.add(VGroup(neurone, etichetta))

        # Strato nascosto (mostra sottoinsieme di 100)
        neuroni_nascosti = VGroup()
        posizioni_y = np.linspace(3, -3, n_neuroni_visualizzati)
        for i in range(n_neuroni_visualizzati):
            neurone = Circle(radius=0.25, color=GREEN, fill_opacity=0.5)
            neurone.move_to(RIGHT * 0 + UP * posizioni_y[i])
            neuroni_nascosti.add(neurone)

        # Neurone output
        neurone_output = Circle(radius=0.3, color=RED, fill_opacity=0.5)
        neurone_output.move_to(RIGHT * 5 + UP * 0)
        etichetta_output = Text("√(a²+b²)", font_size=20)
        etichetta_output.move_to(neurone_output.get_center())
        gruppo_output = VGroup(neurone_output, etichetta_output)

        # Disegna neuroni
        self.play(
            *[FadeIn(inp) for inp in neuroni_input],
            run_time=0.5
        )
        self.play(
            *[FadeIn(h) for h in neuroni_nascosti],
            run_time=0.5
        )
        self.play(FadeIn(gruppo_output), run_time=0.5)

        # Indici campione da visualizzare dai neuroni nascosti
        indici_visualizzazione = np.linspace(0, n_neuroni_nascosti - 1, n_neuroni_visualizzati, dtype=int)

        # Crea connessioni da input a strato nascosto con etichette peso
        connessioni_input_nascosto = VGroup()
        etichette_peso_ih = VGroup()

        for i in range(2):
            for j, h_idx in enumerate(indici_visualizzazione):
                peso = W1[i, h_idx]
                # Normalizza peso per colore (assumendo pesi approssimativamente in range [-1, 1])
                peso_normalizzato = np.clip(peso / 2, -1, 1)

                if peso_normalizzato > 0:
                    colore = interpolate_color(WHITE, RED, abs(peso_normalizzato))
                else:
                    colore = interpolate_color(WHITE, BLUE, abs(peso_normalizzato))

                linea = Line(
                    neuroni_input[i][0].get_center(),
                    neuroni_nascosti[j].get_center(),
                    stroke_width=abs(peso) * 5,
                    color=colore,
                    stroke_opacity=0.6
                )
                connessioni_input_nascosto.add(linea)

                # Aggiungi etichetta valore peso sulla connessione
                testo_peso = Text(f"{peso:.2f}", font_size=10, color=colore)
                testo_peso.move_to(linea.get_center())
                testo_peso.rotate(linea.get_angle())
                etichette_peso_ih.add(testo_peso)

        # Crea connessioni da strato nascosto a output con etichette peso
        connessioni_nascosto_output = VGroup()
        etichette_peso_ho = VGroup()

        for j, h_idx in enumerate(indici_visualizzazione):
            peso = W2[h_idx, 0]
            peso_normalizzato = np.clip(peso / 2, -1, 1)

            if peso_normalizzato > 0:
                colore = interpolate_color(WHITE, RED, abs(peso_normalizzato))
            else:
                colore = interpolate_color(WHITE, BLUE, abs(peso_normalizzato))

            linea = Line(
                neuroni_nascosti[j].get_center(),
                neurone_output.get_center(),
                stroke_width=abs(peso) * 5,
                color=colore,
                stroke_opacity=0.6
            )
            connessioni_nascosto_output.add(linea)

            # Aggiungi etichetta valore peso sulla connessione
            testo_peso = Text(f"{peso:.2f}", font_size=10, color=colore)
            testo_peso.move_to(linea.get_center())
            testo_peso.rotate(linea.get_angle())
            etichette_peso_ho.add(testo_peso)

        # Anima connessioni
        self.play(
            *[Create(conn) for conn in connessioni_input_nascosto],
            run_time=2
        )
        self.play(
            *[FadeIn(etichetta) for etichetta in etichette_peso_ih],
            run_time=1
        )
        self.play(
            *[Create(conn) for conn in connessioni_nascosto_output],
            run_time=2
        )
        self.play(
            *[FadeIn(etichetta) for etichetta in etichette_peso_ho],
            run_time=1
        )

        # Aggiungi leggenda
        leggenda = VGroup()
        titolo_leggenda = Text("Colori Pesi:", font_size=20)
        leggenda_pos = Text("Rosso = Positivo", font_size=18, color=RED)
        leggenda_neg = Text("Blu = Negativo", font_size=18, color=BLUE)
        leggenda_spessore = Text("Spessore = Grandezza", font_size=18)

        leggenda.add(titolo_leggenda, leggenda_pos, leggenda_neg, leggenda_spessore)
        leggenda.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        leggenda.to_corner(DL)

        self.play(FadeIn(leggenda))

        # Aggiungi nota sulla vista semplificata
        if n_neuroni_visualizzati < n_neuroni_nascosti:
            nota = Text(
                f"Mostrando {n_neuroni_visualizzati} di {n_neuroni_nascosti} neuroni nascosti",
                font_size=18,
                color=YELLOW
            )
            nota.to_corner(DR)
            self.play(FadeIn(nota))
        else:
            nota = Text(
                f"Mostrando tutti i {n_neuroni_nascosti} neuroni nascosti",
                font_size=18,
                color=YELLOW
            )
            nota.to_corner(DR)
            self.play(FadeIn(nota))

        self.wait(2)

        # Mostra statistiche pesi
        statistiche = VGroup()
        w1_media = Text(f"W1 media: {np.mean(W1):.4f}", font_size=20)
        w1_dev_std = Text(f"W1 dev std: {np.std(W1):.4f}", font_size=20)
        w2_media = Text(f"W2 media: {np.mean(W2):.4f}", font_size=20)
        w2_dev_std = Text(f"W2 dev std: {np.std(W2):.4f}", font_size=20)

        statistiche.add(w1_media, w1_dev_std, w2_media, w2_dev_std)
        statistiche.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        statistiche.to_corner(UR)

        self.play(FadeIn(statistiche))
        self.wait(3)

        # Sfuma tutto
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )


class HeatmapPesi(Scene):
    def construct(self):
        # Carica pesi modello addestrato
        try:
            with open('model_weights.pkl', 'rb') as f:
                dati_pesi = pickle.load(f)
            W1 = dati_pesi['W1']  # Forma: (2, 100)
            W2 = dati_pesi['W2']  # Forma: (100, 1)
        except FileNotFoundError:
            self.add(Text("Errore: model_weights.pkl non trovato!", color=RED))
            return

        # Ottieni dimensione rete reale
        n_neuroni_nascosti = W1.shape[1]

        # Titolo
        titolo = Text("Mappe di Calore Pesi", font_size=40)
        titolo.to_edge(UP)
        self.play(Write(titolo))
        self.wait(0.5)

        # Crea heatmap per W1
        titolo_w1 = Text(f"Pesi Input → Strato Nascosto (2×{n_neuroni_nascosti})", font_size=24)
        titolo_w1.move_to(UP * 2.5 + LEFT * 3)

        # Normalizza W1 per visualizzazione
        W1_norm = (W1 - W1.min()) / (W1.max() - W1.min())

        # Crea griglia per W1 (mostra come 2 righe, n_nascosti colonne)
        n_display_w1 = min(n_neuroni_nascosti, 50)  # Limita a 50 per visibilità
        larghezza_cella = min(0.1, 5.0 / n_display_w1)  # Aggiusta larghezza in base al conteggio
        altezza_cella = 0.3
        griglia_w1 = VGroup()

        for i in range(2):
            for j in range(n_display_w1):
                valore_colore = W1_norm[i, j]
                colore = interpolate_color(BLUE, RED, valore_colore)

                rettangolo = Rectangle(
                    width=larghezza_cella,
                    height=altezza_cella,
                    fill_color=colore,
                    fill_opacity=0.8,
                    stroke_width=0.5
                )
                rettangolo.move_to(
                    LEFT * 3 +
                    RIGHT * (j * larghezza_cella - n_display_w1 * larghezza_cella / 2) +
                    UP * (0.5 - i * altezza_cella)
                )
                griglia_w1.add(rettangolo)

        self.play(Write(titolo_w1))
        self.play(FadeIn(griglia_w1), run_time=1.5)

        # Crea heatmap per W2
        titolo_w2 = Text(f"Pesi Nascosto → Output ({n_neuroni_nascosti}×1)", font_size=24)
        titolo_w2.move_to(UP * 2.5 + RIGHT * 3)

        # Normalizza W2 per visualizzazione
        W2_norm = (W2 - W2.min()) / (W2.max() - W2.min())

        # Crea griglia per W2 (organizza in pattern griglia)
        griglia_w2 = VGroup()

        # Calcola dimensioni griglia (prova per layout quasi-quadrato)
        colonne_griglia = int(np.ceil(np.sqrt(n_neuroni_nascosti)))
        righe_griglia = int(np.ceil(n_neuroni_nascosti / colonne_griglia))

        cell_width2 = min(0.1, 2.5 / colonne_griglia)
        cell_height2 = min(0.1, 2.5 / righe_griglia)

        for i in range(n_neuroni_nascosti):
            color_value = W2_norm[i, 0]
            color = interpolate_color(BLUE, RED, color_value)

            rect = Rectangle(
                width=cell_width2,
                height=cell_height2,
                fill_color=color,
                fill_opacity=0.8,
                stroke_width=0.3
            )
            # Arrange in grid
            row = i // colonne_griglia
            col = i % righe_griglia
            rect.move_to(
                RIGHT * 3 +
                RIGHT * (col * cell_width2 - colonne_griglia * cell_width2 / 2) +
                UP * (righe_griglia * cell_height2 / 2 - row * cell_height2)
            )
            griglia_w2.add(rect)

        self.play(Write(titolo_w2))
        self.play(FadeIn(griglia_w2), run_time=1.5)

        # Add color scale legend
        legend_title = Text("Color Scale:", font_size=20)
        legend_title.to_corner(DL).shift(UP * 0.5)

        # Create color bar
        color_bar = VGroup()
        for i in range(20):
            color_value = i / 19
            color = interpolate_color(BLUE, RED, color_value)
            segment = Rectangle(
                width=0.3,
                height=0.1,
                fill_color=color,
                fill_opacity=0.8,
                stroke_width=0.5
            )
            segment.move_to(LEFT * 5.5 + RIGHT * (i * 0.3) + DOWN * 2.5)
            color_bar.add(segment)

        min_label = Text("Min", font_size=16).next_to(color_bar[0], DOWN)
        max_label = Text("Max", font_size=16).next_to(color_bar[-1], DOWN)

        self.play(Write(legend_title))
        self.play(FadeIn(color_bar), FadeIn(min_label), FadeIn(max_label))

        self.wait(3)


class CalcoloEsempio(Scene):
    def construct(self):
        # Carica pesi modello addestrato
        try:
            with open('model_weights.pkl', 'rb') as f:
                dati_pesi = pickle.load(f)
            W1 = dati_pesi['W1']  # Forma: (2, n_nascosti)
            b1 = dati_pesi['b1']  # Forma: (n_nascosti,)
            W2 = dati_pesi['W2']  # Forma: (n_nascosti, 1)
            b2 = dati_pesi['b2']  # Forma: (1,)
        except FileNotFoundError:
            self.add(Text("Errore: model_weights.pkl non trovato!", color=RED))
            return

        n_hidden = W1.shape[1]

        # Titolo
        titolo = Text("Calcolo Esempio: f(1, 2)", font_size=36)
        titolo.to_edge(UP)
        self.play(Write(titolo))
        self.wait(0.5)

        # Mostra output atteso
        atteso = np.sqrt(1**2 + 2**2)
        testo_atteso = Text(f"Atteso: √(1² + 2²) = √5 ≈ {atteso:.4f}", font_size=24, color=YELLOW)
        testo_atteso.next_to(titolo, DOWN)
        self.play(FadeIn(testo_atteso))
        self.wait(1)

        # Passo 1: Mostra input
        titolo_passo1 = Text("Passo 1: Strato Input", font_size=28, color=BLUE)
        titolo_passo1.move_to(UP * 2)

        eq_input = MathTex(r"x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}", font_size=40)
        eq_input.move_to(UP * 1)

        self.play(Write(titolo_passo1))
        self.play(Write(eq_input))
        self.wait(1.5)

        # Passo 2: Mostra matrici pesi e bias dello strato nascosto
        self.play(FadeOut(titolo_passo1), FadeOut(eq_input))

        titolo_passo2 = Text("Passo 2: Pesi e Bias Strato Nascosto", font_size=28, color=GREEN)
        titolo_passo2.to_edge(UP)
        self.play(Write(titolo_passo2))
        self.wait(0.5)

        # Calcola output per dopo (ma non mostrare ancora)
        x_input = np.array([1, 2])
        z_hidden = W1.T @ x_input + b1
        a_hidden = np.maximum(0, z_hidden)  # ReLU activation

        # Crea rappresentazione matrice W1 (2 × n_hidden)
        titolo_w1 = Text(f"Matrice W1 (2 × {n_hidden})", font_size=24)
        titolo_w1.move_to(UP * 2.5 + LEFT * 3)

        # Costruisci la matrice W1 come testo matematico
        max_show_cols = 8  # Mostra massimo 8 colonne per leggibilità
        if n_hidden <= max_show_cols:
            # Mostra tutta la matrice
            w1_rows = []
            for i in range(2):
                row_vals = [f"{W1[i,j]:.2f}" for j in range(n_hidden)]
                w1_rows.append(" & ".join(row_vals))
            w1_matrix_str = r"\begin{bmatrix}" + r" \\ ".join(w1_rows) + r"\end{bmatrix}"
        else:
            # Mostra prime colonne + ... + ultime colonne
            show_first = 4
            show_last = 2
            w1_rows = []
            for i in range(2):
                row_vals = [f"{W1[i,j]:.2f}" for j in range(show_first)]
                row_vals.append(r"\cdots")
                row_vals.extend([f"{W1[i,j]:.2f}" for j in range(n_hidden-show_last, n_hidden)])
                w1_rows.append(" & ".join(row_vals))
            w1_matrix_str = r"\begin{bmatrix}" + r" \\ ".join(w1_rows) + r"\end{bmatrix}"

        w1_matrix = MathTex(w1_matrix_str, font_size=20)
        w1_matrix.next_to(titolo_w1, DOWN, buff=0.3)
        w1_matrix.shift(LEFT * 3)

        # Crea rappresentazione vettore b1
        titolo_b1 = Text(f"Vettore b1 ({n_hidden})", font_size=24)
        titolo_b1.move_to(UP * 2.5 + RIGHT * 3)

        if n_hidden <= max_show_cols:
            b1_vals = [f"{b1[j]:.2f}" for j in range(n_hidden)]
            b1_vector_str = r"\begin{bmatrix}" + r" \\ ".join(b1_vals) + r"\end{bmatrix}"
        else:
            b1_vals = [f"{b1[j]:.2f}" for j in range(show_first)]
            b1_vals.append(r"\vdots")
            b1_vals.extend([f"{b1[j]:.2f}" for j in range(n_hidden-show_last, n_hidden)])
            b1_vector_str = r"\begin{bmatrix}" + r" \\ ".join(b1_vals) + r"\end{bmatrix}"

        b1_vector = MathTex(b1_vector_str, font_size=20)
        b1_vector.next_to(titolo_b1, DOWN, buff=0.3)
        b1_vector.shift(RIGHT * 3)

        self.play(Write(titolo_w1), Write(titolo_b1))
        self.play(FadeIn(w1_matrix), FadeIn(b1_vector))
        self.wait(2)

        # Passo 3: Mostra matrici pesi e bias dello strato output
        self.play(
            FadeOut(titolo_passo2), FadeOut(titolo_w1), FadeOut(titolo_b1),
            FadeOut(w1_matrix), FadeOut(b1_vector)
        )

        titolo_passo3 = Text("Passo 3: Pesi e Bias Strato Output", font_size=28, color=RED)
        titolo_passo3.to_edge(UP)
        self.play(Write(titolo_passo3))
        self.wait(0.5)

        # Calcola output
        y_output = W2.T @ a_hidden + b2
        prediction = y_output[0]

        # Crea rappresentazione matrice W2 (n_hidden × 1)
        titolo_w2 = Text(f"Matrice W2 ({n_hidden} × 1)", font_size=24)
        titolo_w2.move_to(UP * 2.5 + LEFT * 3)

        if n_hidden <= max_show_cols:
            w2_vals = [f"{W2[j,0]:.2f}" for j in range(n_hidden)]
            w2_matrix_str = r"\begin{bmatrix}" + r" \\ ".join(w2_vals) + r"\end{bmatrix}"
        else:
            w2_vals = [f"{W2[j,0]:.2f}" for j in range(show_first)]
            w2_vals.append(r"\vdots")
            w2_vals.extend([f"{W2[j,0]:.2f}" for j in range(n_hidden-show_last, n_hidden)])
            w2_matrix_str = r"\begin{bmatrix}" + r" \\ ".join(w2_vals) + r"\end{bmatrix}"

        w2_matrix = MathTex(w2_matrix_str, font_size=20)
        w2_matrix.next_to(titolo_w2, DOWN, buff=0.3)
        w2_matrix.shift(LEFT * 3)

        # Crea rappresentazione scalare b2
        titolo_b2 = Text("Scalare b2", font_size=24)
        titolo_b2.move_to(UP * 2.5 + RIGHT * 3)

        b2_scalar = MathTex(f"{b2[0]:.4f}", font_size=30)
        b2_scalar.next_to(titolo_b2, DOWN, buff=0.3)
        b2_scalar.shift(RIGHT * 3)

        self.play(Write(titolo_w2), Write(titolo_b2))
        self.play(FadeIn(w2_matrix), FadeIn(b2_scalar))
        self.wait(2)

        # Mostra risultato finale
        self.play(
            FadeOut(titolo_passo3), FadeOut(titolo_w2), FadeOut(titolo_b2),
            FadeOut(w2_matrix), FadeOut(b2_scalar)
        )

        titolo_risultato = Text("Risultato Finale", font_size=32, color=PURPLE)
        titolo_risultato.to_edge(UP)

        linea_pred = Text(f"Predetto:  {prediction:.4f}", font_size=28)
        linea_reale = Text(f"Reale:     {atteso:.4f}", font_size=28)
        errore = abs(prediction - atteso)
        errore_pct = (errore / atteso) * 100
        linea_errore = Text(f"Errore:      {errore:.4f} ({errore_pct:.2f}%)", font_size=28,
                         color=GREEN if errore_pct < 5 else YELLOW if errore_pct < 10 else RED)

        risultato = VGroup(linea_pred, linea_reale, linea_errore)
        risultato.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        risultato.move_to(ORIGIN)

        self.play(Write(titolo_risultato))
        self.play(*[Write(linea) for linea in risultato])
        self.wait(3)


if __name__ == "__main__":
    # Questo permette l'esecuzione con: python animate_weights.py
    pass
