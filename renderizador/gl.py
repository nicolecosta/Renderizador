#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import numpy as np
import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante

    stack = [np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])]

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""

        #organizando o RGB
        R = round(colors['emissiveColor'][0]*255,0)
        G = round(colors['emissiveColor'][1]*255,0)
        B = round(colors['emissiveColor'][2]*255,0)

        #separando e desenhando os pontos
        for i in range(0,len(point),2):
            x = int(point[i])
            y = int(point[i+1])
            #gpu.GPU.set_pixel(x, y, R, G, B)
            gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, [R, G, B])  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)
        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""

        #organizando o RGB
        R = round(colors['emissiveColor'][0]*255,0)
        G = round(colors['emissiveColor'][1]*255,0)
        B = round(colors['emissiveColor'][2]*255,0)

        #separando e nomeando os pontos
        for i in range(0,len(lineSegments)-2,2):
            x0 = int(lineSegments[i])
            x1 = int(lineSegments[i+2])
            y0 = int(lineSegments[i+1])
            y1 = int(lineSegments[i+3])
            dx = abs(x1-x0) #delta x
            dy = abs(y1-y0) #delta y
            sx = 1 if x0 < x1 else -1 #identifica a direção da linha
            sy = 1 if y0 < y1 else -1 #identifica a direção da linha
            erro = dx - dy

            #utilizando os princípios de erro incremental de Bresenham
            #assim o código funciona para todos os octantes
            #referência: https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm#:~:text=Bresenham's%20line%20algorithm%20is%20a,straight%20line%20between%20two%20points.
            while x0 != x1 or y0 != y1:
                if x0>= 0 and y0>= 0 and x0<GL.width and y0<GL.height: #limitar linhas dentro do FrameBuffer
                    #gpu.GPU.draw_pixel(x0, y0, R, G, B)
                    gpu.GPU.draw_pixel([x0, y0], gpu.GPU.RGB8, [R, G, B])
                e2 = 2 * erro
                if e2 > -dy:
                    erro -= dy
                    x0 += sx
                if e2 < dx:
                    erro += dx
                    y0 += sy
            
            if x0>=0 and y0>=0 and x0<GL.width and y0<GL.height: #limitar linhas dentro do FrameBuffer
                #gpu.GPU.set_pixel(x0, y0, R, G, B)
                gpu.GPU.draw_pixel([x0, y0], gpu.GPU.RGB8, [R, G, B])  # altera pixel (u, v, tipo, r, g, b)


    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""

        GL.draw_triangle(vertices,colors)

        # #organizando o RGB
        # R = round(colors['emissiveColor'][0]*255,0)
        # G = round(colors['emissiveColor'][1]*255,0)
        # B = round(colors['emissiveColor'][2]*255,0)

        # #separando e nomeando os pontos
        # for i in range(0,len(vertices),6): 
        #     x0 = int(vertices[i])
        #     y0 = int(vertices[i+1])
        #     x1 = int(vertices[i+2])
        #     y1 = int(vertices[i+3])
        #     x2 = int(vertices[i+4])
        #     y2 = int(vertices[i+5])

        #     GL.polyline2D([x0,y0,x1,y1], colors)
        #     GL.polyline2D([x1,y1,x2,y2], colors)
        #     GL.polyline2D([x2,y2,x0,y0], colors)
            

        #     #pegando o max e min para delimitar uma bounding box
        #     max_x = max(x0,x1,x2)
        #     max_y = max(y0,y1,y2)
        #     min_x = min(x0,x1,x2)
        #     min_y = min(y0,y1,y2)

        #     #passando nos pixeis e coloring os que estão dentro dos triângulos
        #     # for x in range(min_x,max_x):
        #     #     for y in range(min_y,max_y):
        #     for x in range(GL.width):
        #         for y in range(GL.height):
        #             L1 = (y1-y0)*x - (x1-x0)*y + y0*(x1-x0) - x0*(y1-y0)
        #             L2 = (y2-y1)*x - (x2-x1)*y + y1*(x2-x1) - x1*(y2-y1)
        #             L3 = (y0-y2)*x - (x0-x2)*y + y2*(x0-x2) - x2*(y0-y2)

        #             if L1 >= 0 and L2 >= 0 and L3 >=0:
        #                 #gpu.GPU.set_pixel(x, y, R, G, B) 
        #                 if x0>=0 and y0>=0 and x0<GL.width and y0<GL.height:
        #                     gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, [R,G,B])  # altera pixel (u, v, tipo, r, g, b)



    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""

        GL.draw_triangle(point,colors, transparecy = True)

        # #separando e nomeando os pontos
        # for i in range(0,len(point),9): 
        #     x0 = (point[i])
        #     y0 = (point[i+1])
        #     z0 = (point[i+2])
        #     x1 = (point[i+3])
        #     y1 = (point[i+4])
        #     z1 = (point[i+5])
        #     x2 = (point[i+6])
        #     y2 = (point[i+7])
        #     z2 = (point[i+8])


        #     M = np.array([[x0, x1, x2],
        #                   [y0, y1, y2],
        #                   [z0, z1, z2],
        #                   [1.0, 1.0, 1.0]])
        #     #print("M= {0}".format(M))
            
    
            
        #     M_T = np.matmul(GL.model, M)
        #     M_T_V = np.matmul(GL.V, M_T)

        #     # print("MxT= {0}".format(M_T))
        #     #print("MxTxV= {0}".format(M_T_V))

        #     last_row = M_T_V[-1]
        #     M_T_V_hom = M_T_V/last_row
        #     #print("HOM= {0}".format(M_T_V_hom))
            
        #     points = []
        #     for i in range(3):
        #         points.append(M_T_V_hom[0][i])
        #         points.append(M_T_V_hom[1][i])

        #     #print("points{0}".format(points))
        #     GL.triangleSet2D(points,colors)


    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.



        #look at
        ux = orientation[0]
        uy = orientation[1]
        uz = orientation[2]
        ang = orientation[3]
        q = np.array([ux*np.sin((ang)/2),uy*np.sin((ang)/2),uz*np.sin((ang)/2),np.cos((ang)/2)])
        q = q/np.linalg.norm(q)
        qi = q[0]
        qj = q[1]
        qk = q[2]
        qr = q[3]
        r11 = 1.0-2.0*(qj**2+qk**2)
        r12 = 2.0*(qi*qj-qk*qr)
        r13 = 2.0*(qi*qk+qj*qr)
        r14 = 0.0
        r21 = 2.0*(qi*qj+qk*qr)
        r22 = 1.0-2.0*(qi**2+qk**2)
        r23 = 2.0*(qj*qk-qi*qr)
        r24 = 0.0
        r31 = 2.0*(qi*qk-qj*qr)
        r32 = 2.0*(qj*qk+qi*qr)
        r33 = 1.0-2.0*(qi**2+qj**2)
        r34 = 0.0
        r41 = 0.0
        r42 = 0.0
        r43 = 0.0
        r44 =1.0


        R = np.array([[r11, r12, r13, r14],
                      [r21, r22, r23, r24],
                      [r31, r32, r33, r34],
                      [r41, r42, r43, r44]])
        
        T_id = np.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        position])
        
        hom = [0.0, 0.0, 0.0, 1.0]

        T = np.append(T_id.transpose(),np.array(([hom])),axis=0)


        lookat = np.linalg.inv(np.matmul(T,R))
        #print("Lookat: {0}".format(lookat))  
        GL.lookat = lookat

        #perspective
        width = GL.width 
        height = GL.height 
        near = GL.near 
        far = GL.far 
        fovy = 2.0*np.arctan(np.tan(fieldOfView/2.0)*height/np.sqrt(height**2+width**2)) #em radiano
        top = near * np.tan(fovy)
        right = top*(width/height)

        P = np.array([[near/right, 0.0, 0.0, 0.0],
                      [0.0, near/top, 0.0, 0.0],
                      [0.0, 0.0, -((far+near)/(far-near)), (-2*far*near)/(far-near)],
                      [0.0, 0.0, -1.0, 0.0]])
        #print("Perspective: {0}".format(P))  
        

        #screen
        S = np.array([[width/2, 0.0, 0.0, width/2],
                [0.0, -(height/2), 0.0, height/2],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]])
        #print("Screen: {0}".format(S))  


        #view = screen x perspective x look at
        GL.V = np.matmul(P,lookat)
        GL.V = np.matmul(S,GL.V)
        #print("Viewpoint Matrix: {0}".format(GL.V))           

        # # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Viewpoint : ", end='')
        # print("position = {0} ".format(position), end='')
        # print("orientation = {0} ".format(orientation), end='')
        # print("fieldOfView = {0} ".format(fieldOfView))

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""

        translation_matrix = np.array([[1.0, 0.0, 0.0, translation[0]],
                                     [0.0, 1.0, 0.0, translation[1]],
                                     [0.0, 0.0, 1.0, translation[2]],
                                     [0.0, 0.0, 0.0, 1.0]])
        
        scale_matrix =np.array([[scale[0], 0.0, 0.0, 0.0],
                                [0.0, scale[1], 0.0, 0.0],
                                [0.0, 0.0, scale[2], 0.0],
                                [0.0, 0.0, 0.0, 1.0]])

        ux = rotation[0]
        uy = rotation[1]
        uz = rotation[2]
        ang = rotation[3]
        q = np.array([ux*np.sin((ang)/2),uy*np.sin((ang)/2),uz*np.sin((ang)/2),np.cos((ang)/2)])
        q = q/np.linalg.norm(q)
        qi = q[0]
        qj = q[1]
        qk = q[2]
        qr = q[3]
        r11 = 1.0-2.0*(qj**2+qk**2)
        r12 = 2.0*(qi*qj-qk*qr)
        r13 = 2.0*(qi*qk+qj*qr)
        r14 = 0.0
        r21 = 2.0*(qi*qj+qk*qr)
        r22 = 1.0-2.0*(qi**2+qk**2)
        r23 = 2.0*(qj*qk-qi*qr)
        r24 = 0.0
        r31 = 2.0*(qi*qk-qj*qr)
        r32 = 2.0*(qj*qk+qi*qr)
        r33 = 1.0-2.0*(qi**2+qj**2)
        r34 = 0.0
        r41 = 0.0
        r42 = 0.0
        r43 = 0.0
        r44 =1.0


        rotation_matrix = np.array([[r11, r12, r13, r14],
                      [r21, r22, r23, r24],
                      [r31, r32, r33, r34],
                      [r41, r42, r43, r44]])
        
        pre_transform = np.matmul(rotation_matrix,scale_matrix)
        transform = np.matmul(translation_matrix,pre_transform)

        GL.stack.append(np.matmul(GL.stack[-1], transform))
        GL.model = GL.stack[-1]

        #print("Model = {0} ".format(GL.model))

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""

        GL.model = GL.stack.pop()

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""

        for vertice in range(0,(stripCount[0]*3) - 6,3):
            #print('VERTICE',vertice)
            index1 = vertice+3
            index2 = vertice+6
            index3 = vertice+9

            if vertice%2 ==0:
                vertices = point[vertice:index3]
            else:
                vertices = point[vertice:index1] + point[index2:index3] + point[index1:index2]


            GL.triangleSet(vertices, colors)



    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.


        i = 2
        clockwise = False

        while index[i] != -1:
            if not clockwise:
                points = [point[index[i-2]*3], point[index[i-2]*3+1], point[index[i-2]*3+2],
                          point[index[i-1]*3], point[index[i-1]*3+1], point[index[i-1]*3+2],
                          point[index[i]*3], point[index[i]*3+1], point[index[i]*3+2]] 
            else:
                points = [point[index[i-2]*3], point[index[i-2]*3+1], point[index[i-2]*3+2],
                          point[index[i]*3], point[index[i]*3+1], point[index[i]*3+2],
                          point[index[i-1]*3], point[index[i-1]*3+1], point[index[i-1]*3+2]] 
                
            clockwise = not clockwise
            i += 1
            GL.triangleSet(points, colors)




    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        #print("Box : size = {0}".format(size)) # imprime no terminal pontos
        #print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        #gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        clockwise = False
        i =2 
        while coordIndex[i] != -1:
            if not clockwise:
                points = [coord[coordIndex[i]*3], coord[coordIndex[i]*3+1], coord[coordIndex[i]*3+2],
                          coord[coordIndex[i+1]*3], coord[coordIndex[i+1]*3+1], coord[coordIndex[i+1]*3+2],
                          coord[coordIndex[i+2]*3], coord[coordIndex[i+2]*3+1], coord[coordIndex[i+2]*3+2]] 
            else:
                points = [coord[coordIndex[i]*3], coord[coordIndex[i]*3+1], coord[coordIndex[i]*3+2],
                          coord[coordIndex[i+2]*3], coord[coordIndex[i+2]*3+1], coord[coordIndex[i+2]*3+2],
                          coord[coordIndex[i+1]*3], coord[coordIndex[i+1]*3+1], coord[coordIndex[i+1]*3+2]]
                
            clockwise = not clockwise
            i += 1
            GL.triangleSet(points, colors)


    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatament_e tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""



    @staticmethod
    def calc_bary(A, B, C, P):
        v0 = B - A
        v1 = C - A
        v2 = P - A

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        denom = dot00 * dot11 - dot01 * dot01
        u = (dot11 * dot02 - dot01 * dot12) / denom
        v = (dot00 * dot12 - dot01 * dot02) / denom
        w = 1 - u - v

        return u, v, w
    
    @staticmethod
    def is_inside(x, y, points):
        for i in range(0,len(points),6): 
            x0 = int(points[i])
            y0 = int(points[i+1])
            x1 = int(points[i+2])
            y1 = int(points[i+3])
            x2 = int(points[i+4])
            y2 = int(points[i+5])

        L1 = (y1-y0)*x - (x1-x0)*y + y0*(x1-x0) - x0*(y1-y0)
        L2 = (y2-y1)*x - (x2-x1)*y + y1*(x2-x1) - x1*(y2-y1)
        L3 = (y0-y2)*x - (x0-x2)*y + y2*(x0-x2) - x2*(y0-y2)

        if L1 >= 0 and L2 >= 0 and L3 >=0:
            return True
        else: 
            return False

    
    
    @staticmethod
    def antialiasing(x, y, points):
        samplingrate = 4
        xs = int(x*samplingrate)
        ys = int(ys*samplingrate)

        _ss = 0

        sampled_points = [element * samplingrate for element in points]

        if GL.is_inside(xs, ys, sampled_points):
            _ss += 1

        if GL.is_inside(xs+1, ys, sampled_points):
            _ss += 1

        if GL.is_inside(xs, ys+1, sampled_points):
            _ss += 1

        if GL.is_inside(xs+1, ys+1, sampled_points):
            _ss += 1

        _ss /= 16

        return _ss

    @staticmethod
    def new_draw_pixel(x,y,points,color,colors,ss=1):
        frame = 0 < x < GL.width and 0 < y < GL.height
        # Separate into A, B, and C coordinate lists
        A = points[0], points[1]
        B = points[2], points[3]
        C = points[4], points[5]

        points_len = len(points)
        if points_len % 6 == 0:
            dim = '2D'
        elif points_len % 9 == 0:
            dim = '3D'

        og_color = gpu.GPU.read_pixel([x,y], gpu.GPU.RGB8)*colors['transparency']

        if frame == True:
            if color != None and dim == '2D':
                u, v, w = GL.calc_bary(A,B,C)
                
                # Cor interpolada
                R, G, B = u*color[:, 0] + v*color[:, 1] + w*color[:, 2]
                R -= R * colors['transparency']
                G -= G* colors['transparency']
                B -= B * colors['transparency']
                new_color = [R, G, B]

                # Combinando as cores
                R, G, B = og_color + new_color

                gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, [R*255, G*255, B*255])
            
            elif color == None and dim == '2D':
                # New Color
                R -= colors['emissiveColor'][0] * colors['transparency']
                G -= colors['emissiveColor'][1] * colors['transparency']
                B -= colors['emissiveColor'][2] * colors['transparency']
                new_color = [R, G, B]

                R, G, B = og_color + new_color

                gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, [R*255*ss, G*255*ss, B*255*ss]) 

            elif color != None and dim == '3D':
                Z = 1/(u/points[2] + v/points[5] + w/points[8])

                if(Z < gpu.GPU.read_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F)):

                    gpu.GPU.draw_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F, [Z])

                    # Cor interpolada (levando em conta deformação da perspectiva)
                    r, g, b = Z*(u*color[:, 0]/points[2] + v*color[:, 1]/points[5] + w*color[:, 2]/points[8])
                    r *= (1-colors['transparency'])*255
                    g *= (1-colors['transparency'])*255
                    b *= (1-colors['transparency'])*255

                    # Seta que as cores estejam no intervalo entre 0 e 255
                    R = max(min(r, 255.0), 0.0)
                    G = max(min(g, 255.0), 0.0)
                    B = max(min(b, 255.0), 0.0)

                    new_color = [R, G, B]

                    # Combinando as cores
                    R, G, B = og_color + new_color

                    gpu.GPU.draw_pixel([x,y], gpu.GPU.RGB8, [R,G,B]) 

            else:
                Z = 1/(u/points[2] + v/points[5] + w/points[8])
                if(Z < gpu.GPU.read_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F)):

                    gpu.GPU.draw_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F, [Z])

                    # New Color
                    R = colors['emissiveColor'][0]*(1-colors['transparency'])*255.0
                    G = colors['emissiveColor'][1]*(1-colors['transparency'])*255.0
                    B = colors['emissiveColor'][2]*(1-colors['transparency'])*255.0
                    new_color = [R, G, B]

                    # Combinando as cores
                    R, G, B = og_color + new_color

                    # r, g, b = color_buffer
                    gpu.GPU.draw_pixel([x,y], gpu.GPU.RGB8, [R,G,B])


     @staticmethod
    def draw_triangle(points, colors, color=None, transparency=False):
        points_len = len(points)
        if points_len % 6 == 0:
            dim = '2D'
            num_coord = 6
        elif points_len % 9 == 0:
            dim = '3D'
            num_coord = 6

        # Pega o total de triângulos e os separa em uma matriz de triângulos
        total_triangles = int(points_len/num_coord)
        total_triangles = total_triangles if total_triangles != 0 else 1
        triangles = np.array_split(points, total_triangles)

        for i in range(total_triangles):
            vertices = triangles[i]
            
            if dim == '3D':
                for i in range(0,len(vertices),9): 
                    x0 = int(vertices[i])
                    y0 = int(vertices[i+1])
                    z0 = int(vertices[i+2])
                    x1 = int(vertices[i+3])
                    y1 = int(vertices[i+4])
                    z1 = int(vertices[i+5])
                    x2 = int(vertices[i+6])
                    y2 = int(vertices[i+7])
                    z2 = int(vertices[i+8])


                # Montando matriz de coordenadas
                M = np.array([[x0, x1, x2],
                                        [x0, y1, y2],
                                        [z0, z1, z2],
                                        [1.0, 1.0, 1.0]])

        
                # Multiplicando por matriz de transform            
                M_T = np.matmul(GL.model, M)

                # Obtendo matriz do Z para a deformação de perspectiva
                temp_Z = np.matmul(GL.lookat, M_T)

                # Multiplicando por matriz de view
                M_T_V = np.matmul(GL.V, M_T)

                last_row = M_T_V[-1]
                M_T_V_hom = M_T_V/last_row
                
                points = []
                z_coord = []
                z_NDC = []
                for i in range(3):
                    points.append(M_T_V_hom[0][i])
                    points.append(M_T_V_hom[1][i])
                    z_NDC.append(M_T_V[2][i])
                    z_coord.append(temp_Z[2][i])
                    

            else:
                # Criando lista de pontos 
                points = []
                points.append(vertices[0])
                points.append(vertices[1])
                points.append(vertices[2])
                points.append(vertices[3])
                points.append(vertices[4])
                points.append(vertices[5])

            x0, y0 = points[0], points[1]
            x1, y1 = points[2], points[3]
            x2, y2 = points[4], points[5]

            if dim == '3D':
                if not transparency:
                    z0, z1, z2 = z_coord[0], z_coord[1], z_coord[2]
                else:
                    z0, z1, z2 = z_NDC[0], z_NDC[1], z_NDC[2]
                passPoints = [x0, y0,z0, x1, y1, z1, x2, y2, z2]
            else:
                passPoints = [x0, y0, x1, y1, x2, y2]

                    
            # Ordem de conexão 
            connectionPoints = [x0, y0, x1, y1, x2, y2, x0, y0]

            for i in range(0, 5, 2):
                x0, y0 = round(connectionPoints[i]), round(connectionPoints[i+1])
                x1, y1 = round(connectionPoints[i+2]), round(connectionPoints[i+3])

                dx = abs(x1-x0) #delta x
                dy = abs(y1-y0) #delta y
                sx = 1 if x0 < x1 else -1 #identifica a direção da linha
                sy = 1 if y0 < y1 else -1 #identifica a direção da linha
                erro = dx - dy
                
                while True:
                    # Desenha pixel (3D/2D)
                    if dim == '3D':
                        GL.new_draw_pixel(x0, y0, passPoints, color, colors)
                    else:
                        # Anti aliasing (apenas para exemplo 2D)
                        _ss = GL.antialiasing(x0, y0, passPoints)
                        GL.new_draw_pixel(x0, y0, passPoints, color, colors, ss=_ss)                        

                    e2 = 2 * erro
                    if e2 > -dy:
                        erro -= dy
                        x0 += sx
                    if e2 < dx:
                        erro += dx
                        y0 += sy

            #pegando o max e min para delimitar uma bounding box
            max_x = max(x0,x1,x2)
            max_y = max(y0,y1,y2)
            min_x = min(x0,x1,x2)
            min_y = min(y0,y1,y2)

            for i in range(min_x, max_x, 1):
                for j in range(min_y, max_y, 1):
                    if GL.is_inside(i, j, passPoints):
                        if dim == '3D':
                            GL.new_draw_pixel(i, j, passPoints, color, colors)
                        else:
                            # Anti aliasing (apenas para exemplo 2D)
                            _ss = GL.antialiasing(i, j, points)
                            GL.new_draw_pixel(i, j, passPoints, color, colors, ss=_ss)


