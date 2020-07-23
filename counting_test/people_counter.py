from counting_test.trackable_object import TrackableObject
from counting_test.centroidtracker import CentroidTracker
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="Path to Caffe 'deploy prototxt file")
ap.add_argument("-m", "--model", required=True, help="Caminho para modelo pre-treinado")
ap.add_argument("-i", "--input", type=str, help="Caminho para arquivo de video opcional")
ap.add_argument("-o", "--output", type=str, help="Caminho para video de saida opcional")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
                help="Probabilidade minima, para filtrar deteccoes fracas")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
                help="# de frames pulados entre cada deteccao")
args = vars(ap.parse_args())

# Inicializando a lista de classes que a rede MobileNet SSD foi treinada para detectar
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Carregar o modelo pre-treinado do disco
print("[INFO] carregando modelo...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Se um video nao foi suprido, utilizar WebCam
if not args.get("input", False):
    print("[INFO] iniciando stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
# C.contrario, processar video
else:
    print("[INFO] abrindo arquivo de video...")
    vs = cv2.VideoCapture(args["input"])

# Inicializando o writer de video, instaciado depois, se necessario
writer = None
# Inicializando as dimensoes de video
W = None
H = None

# Instanciar o rastreador de centroides
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
# Inicializar uma lista para guardar cada correlation tracker da dlib
trackers = []
# Inicializar um dicionario para associar cada ID de objeto a um TrackableObject
trackableObjects = {}

# Inicializar o numero de frames processados ate entao e o numero de objetos que se
# moveram para cima ou para baixo
totalFrames = 0
totalDown = 0
totalUp = 0

# Inicializar o estimador de frames por segundo
fps = FPS().start()

# Comecar a ler os frames
while True:
    # Pegar o proximo frame e tratar se e proveniente de VideoCapture ou VideoStream
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame
    # Se estamos processando um video e nao recebemos um frame, chegamos ao fim
    if args["input"] is not None and frame is None:
        break

    # Redimensionar o frame para uma largura de no maximo 500 pixels e converter o frame
    # de BGR (formato de leitura da OpenCV) para RGB (formato de leitura de frame da DLib)
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Se as dimensoes do frame estiverem vazias, instancie-as
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Se tivermos que escrever um video em disco, instanciar o writer
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

    # Inicializar o status e a lista de Bounding Boxes retornadas pelo (1) detector de
    # objetos, ou (2) pelos filtros de correlacao
    status = "Waiting"
    rects = []

    # Checa se esta na hora de rodar o detector (computacionalmente mais caro) para
    # ajudar no rastreio
    if totalFrames % args["skip_frames"] == 0:
        # Setar status e inicializar uma nova lista de trackers
        status = "Detecting"
        trackers = []

        # Converter o frame para um blob e passa-lo pela rede para obter as detecções
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Processar as deteccoes provenientes do modelo
        for i in np.arange(0, detections.shape[2]):
            # Extrair a confianca associada a predicao
            confidence = detections[0, 0, i, 2]
            # Filtrar deteccoes fracas utilizando um valor de confianca minimo
            if confidence > args["confidence"]:
                # Extrair o indice da classe da lista de deteccoes
                idx = int(detections[0, 0, i, 1])
                # Se nao for da classe pessoa, nao interessa
                if CLASSES[idx] != "person":
                    continue

                # Computar as coordenadas (x, y) da B.Box para o objeto
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                # Construir um objeto retangulo da Dlib a partir das coordenadas da B.Box
                # e iniciar os filtros de correlacao da biblioteca Dlib
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # Adicionar o rastreador a lista de rastreadores para que possa ser
                # utilizado durante os frames de skip (quando nao rodamos o detector)
                trackers.append(tracker)
    # C.contrario, devemos utilizar os filtros de correlacao ao inves do detector
    else:
        # Para cada filtro de correlacao rastreador
        for tracker in trackers:
            # Mudar status para 'tracking' ou 'rastreando'
            status = "Tracking"
            # Atualiza o rastreador (tracker) e pega a posicao atualizada
            tracker.update(rgb)
            pos = tracker.get_position()
            # "desembrulhar" o objeto 'posicao'
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            # Adicionar as coordenadas da nova B.Box na lista
            rects.append((startX, startY, endX, endY))

    # Desenha uma linha horizontal no centro do frame, quando um objeto cruzar essa linha
    # vamos determinar se o mesmo se moveu para cima ou para baixo
    cv2. line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

    # Utilizar o rastreador de centroide para associar o centroide de objetos antigos ao
    # de objetos recentemente computados
    objects = ct.update(rects)

    # Para cada objeto rastreado
    for (objectID, centroid) in objects.items():
        # Checar se um 'trackable object' existe para o ID
        to = trackableObjects.get(objectID, None)

        # Se nao existe um 'trackable object', crie um
        if to is None:
            to = TrackableObject(objectID, centroid)

        # C.C, existe um 'trackable object' e podemos utiliza-lo para determinar direcao
        else:
            # A diferenca a coordenada Y do centroide 'atual' e a media dos centroides
            # anteriores nos dira em qual direcao o objeto esta se movendo (negativo
            # para cima e positivo para baixo)
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # Checar se o objeto foi contado ou nao
            if not to.counted:
                # se a direcao e negativa (indicando que o objeto esta se movendo para
                # cima) E o centroide esta acima do centro da linha, contar o objeto
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    to.counted = True

                # Se a direcao e positiva (indicando que o objeto esta se movendo para
                # baixo) E o centroide esta abaixo da linha, contar o objeto
                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    to.counted = True
        # Guardar o 'trackable object' no dicionario
        trackableObjects[objectID] = to

        # Desenhar o ID do objeto e seu centroide
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # Construir a tupla de informacoes que serao mostradas no frame
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
        ("Status", status)
    ]

    # Desenhar a tupla de informacoes no frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Checar se e preciso escrever o frame no disco
    if writer is not None:
        writer.write(frame)
    # Renderizar o frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Se a tecla 'q' foi pressionada, sair do loop
    if key == ord("q"):
        break

    # Incrementar o total de frames processados e atualizar o contador de FPS
    totalFrames += 1
    fps.update()

# Parar o contador e mostar informacoes de FPS
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[INFO] total frames: {:.2f}".format(totalFrames))

# Checar se e necessario liberar o ponteiro para o escritor de video
if writer is not None:
    writer.release()

# Se nao estamos utilizando um arquivo de video, parar a camera
if not args.get("input", False):
    vs.stop()
# C.C, liberar o ponteiro do video
else:
    vs.release()

# Fechar qualquer janela remanescente
cv2.destroyAllWindows()
