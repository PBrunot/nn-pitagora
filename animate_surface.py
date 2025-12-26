from manim import *
import numpy as np
import pickle

class SuperficieNNEvoluzioneSemplice(ThreeDScene):
    """
    Anima l'apprendimento della rete neurale per approssimare f(a,b) = sqrt(a^2 + b^2)
    Mostra progressione attraverso epoche 1, 3, 10, 30, 50 con animazione morphing fluida.
    """

    def construct(self):
        # Carica dati epoca
        with open('superfici_epoche.pkl', 'rb') as f:
            dati = pickle.load(f)

        range_a = dati['a_range']
        range_b = dati['b_range']
        Z_reale = dati['Z_actual']
        tutte_epoche = sorted(dati['epochs'].keys())

        # Seleziona epoche specifiche da mostrare: 1, 3, 10, 30, 50
        epoche_target = [1, 3, 10, 30, 50]
        # Filtra per includere solo epoche che esistono nei dati
        epoche = [e for e in epoche_target if e in tutte_epoche]

        # Scala per visualizzazione - scala più grande riempie meglio lo schermo
        scala_xy = 0.6
        scala_z = 0.6  # Stessa scala per tutti gli assi

        # Imposta camera per visualizzazione ottimale
        # phi=60 dà un angolo più basso che enfatizza la curvatura della superficie
        # theta=-110 fornisce una vista simile al riferimento screenshot
        # Distanza è impostata più vicino per riempire di più lo schermo
        self.set_camera_orientation(
            phi=60 * DEGREES,
            theta=-110 * DEGREES,
            distance=8  # Camera più vicina per vista più grande
        )
        self.begin_ambient_camera_rotation(rate=0.06)  # Rotazione più lenta per vista migliore

        # Aggiungi assi con scala-z diversa, range ottimizzati
        assi = ThreeDAxes(
            x_range=[-5 * scala_xy, 5 * scala_xy, 2 * scala_xy],
            y_range=[-5 * scala_xy, 5 * scala_xy, 2 * scala_xy],
            z_range=[0, 7 * scala_z, 2 * scala_z],  # Aggiustato per miglior adattamento (max ~7.07 agli angoli)
            x_length=10 * scala_xy,
            y_length=10 * scala_xy,
            z_length=7 * scala_z
        )

        etichette = assi.get_axis_labels(
            x_label=MathTex("a").scale(0.8),
            y_label=MathTex("b").scale(0.8),
            z_label=MathTex("f(a,b)=\\sqrt{a^2+b^2}").scale(0.7)
        )

        # Titolo
        titolo = Text("Apprendimento NN: Teorema di Pitagora", font_size=36)
        titolo.to_corner(UP)

        self.add_fixed_in_frame_mobjects(titolo)
        self.play(Create(assi), Write(etichette), Write(titolo))

        # Crea superficie reale (wireframe blu) con z amplificato
        # Usa i range di dati per assicurare allineamento con superficie NN
        a_min, a_max = range_a[0], range_a[-1]
        b_min, b_max = range_b[0], range_b[-1]

        def funzione_reale(u, v):
            a = a_min + u * (a_max - a_min)
            b = b_min + v * (b_max - b_min)
            z = np.sqrt(a**2 + b**2)
            return np.array([a * scala_xy, b * scala_xy, z * scala_z])

        superficie_reale = Surface(
            funzione_reale,
            u_range=[0, 1],
            v_range=[0, 1],
            resolution=(30, 30),  # Risoluzione più alta per superficie più fluida
            fill_opacity=1.0,  # Riempimento solido per superficie target
            stroke_width=0,  # Nessuna linea griglia - solo superficie fluida
            fill_color=BLUE_C,  # Riempimento blu ricco
            shade_in_3d=True,
            checkerboard_colors=[BLUE_C,BLUE_C]
        )

        self.play(Create(superficie_reale), run_time=2)
        self.wait(2)

        # Crea superfici per epoche selezionate
        superfici = []
        for epoca in epoche:
            Z_pred = dati['epochs'][epoca]

            # Interpola superficie con z amplificato
            def crea_funzione_superficie(Z_dati, a_r, b_r, z_scl):
                def funz_superficie(u, v):
                    # Interpolazione bilineare
                    x_idx = u * (len(a_r) - 1)
                    y_idx = v * (len(b_r) - 1)

                    x0, x1 = int(x_idx), min(int(x_idx) + 1, len(a_r) - 1)
                    y0, y1 = int(y_idx), min(int(y_idx) + 1, len(b_r) - 1)

                    fx, fy = x_idx - x0, y_idx - y0

                    z = (1-fx)*(1-fy)*Z_dati[y0,x0] + fx*(1-fy)*Z_dati[y0,x1] + \
                        (1-fx)*fy*Z_dati[y1,x0] + fx*fy*Z_dati[y1,x1]

                    # Usa range dati reali da a_r e b_r
                    a = a_r[0] + u * (a_r[-1] - a_r[0])
                    b = b_r[0] + v * (b_r[-1] - b_r[0])
                    return np.array([a * scala_xy, b * scala_xy, z * z_scl])
                return funz_superficie

            # Interpolazione colore da rosso a verde basata su progresso
            progresso_colore = epoche.index(epoca) / (len(epoche) - 1) if len(epoche) > 1 else 0
            colore_superficie = interpolate_color(RED, GREEN, progresso_colore)

            superficie = Surface(
                crea_funzione_superficie(Z_pred, range_a, range_b, scala_z),
                u_range=[0, 1],
                v_range=[0, 1],
                resolution=(30, 30),  # Risoluzione più alta per superficie più fluida
                fill_opacity=0.75,  # Leggermente più opaco per risaltare
                stroke_width=0.8,  # Wireframe visibile
                stroke_color=colore_superficie,
                fill_color=colore_superficie,
                stroke_opacity=0.8
            )
            superfici.append((epoca, superficie))

        # Anima transizioni
        superficie_corrente = superfici[0][1]
        etichetta_epoca = Text(f"Epoca {superfici[0][0]}", font_size=32, color=YELLOW).to_corner(UP + LEFT)

        self.add_fixed_in_frame_mobjects(etichetta_epoca)
        self.play(Create(superficie_corrente), Write(etichetta_epoca))
        self.wait(2)  # Pausa più lunga per vedere approssimazione iniziale scadente

        for i in range(1, len(superfici)):
            epoca, superficie_prossima = superfici[i]
            nuova_etichetta = Text(f"Epoca {epoca}", font_size=32, color=YELLOW).to_corner(UP + LEFT)

            # Rimuovi vecchia etichetta da oggetti frame fissi e sfumala
            self.remove(etichetta_epoca)

            # Aggiungi nuova etichetta e anima trasformazione
            self.add_fixed_in_frame_mobjects(nuova_etichetta)
            self.play(
                Transform(superficie_corrente, superficie_prossima),
                FadeOut(etichetta_epoca),
                FadeIn(nuova_etichetta),
                run_time=3  # Leggermente più lungo per apprezzare il morphing
            )

            # Aggiorna riferimento etichetta per prossima iterazione
            etichetta_epoca = nuova_etichetta

            self.wait(2)  # Pausa per vedere risultato di ogni epoca

        # Messaggio finale
        testo_finale = Text("Addestramento Completato!", font_size=24, color=GREEN)
        testo_finale.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(testo_finale)
        self.play(Write(testo_finale))

        self.stop_ambient_camera_rotation()
        self.wait(3)
