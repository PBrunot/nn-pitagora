from manim import *
import numpy as np
import pickle

class SuperficieNNEvoluzioneSemplice(ThreeDScene):
    """
    Anima l'apprendimento della rete neurale per approssimare f(a,b) = sqrt(a^2 + b^2)
    Mostra progressione attraverso epoche 1, 3, 10, 30, 75 con animazione morphing fluida.
    """

    def construct(self):
        # Carica dati epoca
        with open('superfici_epoche.pkl', 'rb') as f:
            dati = pickle.load(f)

        range_a = dati['a_range']
        range_b = dati['b_range']
        Z_reale = dati['Z_actual']
        tutte_epoche = sorted(dati['epochs'].keys())

        # Seleziona epoche specifiche da mostrare
        epoche_target = [1, 3, 10, 30, 75]
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
            # Prima di transizione, crea volume 3D dello spazio di errore tra le superfici
            epoca_corrente = superfici[i-1][0]
            Z_corrente = dati['epochs'][epoca_corrente]

            # Crea gruppo per contenere tutte le superfici che formano il volume
            volume_errore = VGroup()

            # Funzione helper per interpolazione bilineare
            def interpola_z(Z_data, u, v, a_r):
                x_idx = u * (len(a_r) - 1)
                y_idx = v * (len(a_r) - 1)

                x0, x1 = int(x_idx), min(int(x_idx) + 1, len(a_r) - 1)
                y0, y1 = int(y_idx), min(int(y_idx) + 1, len(a_r) - 1)

                fx, fy = x_idx - x0, y_idx - y0

                return (1-fx)*(1-fy)*Z_data[y0,x0] + fx*(1-fy)*Z_data[y0,x1] + \
                       (1-fx)*fy*Z_data[y1,x0] + fx*fy*Z_data[y1,x1]

            # Superficie superiore (quella reale o quella predetta, a seconda di quale è più alta)
            def superficie_sup(u, v):
                z_pred = interpola_z(Z_corrente, u, v, range_a)
                z_real = interpola_z(Z_reale, u, v, range_a)
                z_max = max(z_pred, z_real)

                a = range_a[0] + u * (range_a[-1] - range_a[0])
                b = range_b[0] + v * (range_b[-1] - range_b[0])
                return np.array([a * scala_xy, b * scala_xy, z_max * scala_z])

            # Superficie inferiore
            def superficie_inf(u, v):
                z_pred = interpola_z(Z_corrente, u, v, range_a)
                z_real = interpola_z(Z_reale, u, v, range_a)
                z_min = min(z_pred, z_real)

                a = range_a[0] + u * (range_a[-1] - range_a[0])
                b = range_b[0] + v * (range_b[-1] - range_b[0])
                return np.array([a * scala_xy, b * scala_xy, z_min * scala_z])

            # Crea superfici superiore e inferiore
            sup = Surface(
                superficie_sup,
                u_range=[0, 1],
                v_range=[0, 1],
                resolution=(20, 20),
                fill_opacity=0.5,
                stroke_width=0,
                fill_color=RED,
                shade_in_3d=True
            )

            inf = Surface(
                superficie_inf,
                u_range=[0, 1],
                v_range=[0, 1],
                resolution=(20, 20),
                fill_opacity=0.5,
                stroke_width=0,
                fill_color=RED,
                shade_in_3d=True
            )

            # Crea superfici laterali per chiudere il volume (4 lati)
            n_punti = 20

            # Lato 1: u=0 (bordo sinistro)
            def lato1(u, v):
                z_pred = interpola_z(Z_corrente, 0, u, range_a)
                z_real = interpola_z(Z_reale, 0, u, range_a)
                z = z_real + v * (z_pred - z_real)

                a = range_a[0]
                b = range_b[0] + u * (range_b[-1] - range_b[0])
                return np.array([a * scala_xy, b * scala_xy, z * scala_z])

            # Lato 2: u=1 (bordo destro)
            def lato2(u, v):
                z_pred = interpola_z(Z_corrente, 1, u, range_a)
                z_real = interpola_z(Z_reale, 1, u, range_a)
                z = z_real + v * (z_pred - z_real)

                a = range_a[-1]
                b = range_b[0] + u * (range_b[-1] - range_b[0])
                return np.array([a * scala_xy, b * scala_xy, z * scala_z])

            # Lato 3: v=0 (bordo anteriore)
            def lato3(u, v):
                z_pred = interpola_z(Z_corrente, u, 0, range_a)
                z_real = interpola_z(Z_reale, u, 0, range_a)
                z = z_real + v * (z_pred - z_real)

                a = range_a[0] + u * (range_a[-1] - range_a[0])
                b = range_b[0]
                return np.array([a * scala_xy, b * scala_xy, z * scala_z])

            # Lato 4: v=1 (bordo posteriore)
            def lato4(u, v):
                z_pred = interpola_z(Z_corrente, u, 1, range_a)
                z_real = interpola_z(Z_reale, u, 1, range_a)
                z = z_real + v * (z_pred - z_real)

                a = range_a[0] + u * (range_a[-1] - range_a[0])
                b = range_b[-1]
                return np.array([a * scala_xy, b * scala_xy, z * scala_z])

            for lato_func in [lato1, lato2, lato3, lato4]:
                lato = Surface(
                    lato_func,
                    u_range=[0, 1],
                    v_range=[0, 1],
                    resolution=(n_punti, 10),
                    fill_opacity=0.4,
                    stroke_width=0,
                    fill_color=RED,
                    shade_in_3d=True
                )
                volume_errore.add(lato)

            volume_errore.add(sup, inf)

            # Mostra volume errore con fade in
            self.play(FadeIn(volume_errore), run_time=1.5)
            self.wait(2)

            # Rimuovi volume errore prima del morphing
            self.play(FadeOut(volume_errore), run_time=0.5)

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
