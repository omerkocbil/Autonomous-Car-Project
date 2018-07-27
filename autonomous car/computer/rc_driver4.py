__author__ = 'omer'

import threading
import SocketServer
import cv2
import numpy as np
import math
from sklearn.model_selection import train_test_split
import sys
import glob
import time

# distance data measured by ultrasonic sensor
araba_data = " "


class NeuralNetwork(object):

    def __init__(self):
        self.model = cv2.ml.ANN_MLP_load('mlp_xml/mlp_otonom_sag.xml')

    def predict(self, samples):
        ret, resp = self.model.predict(np.float32(samples))
        return resp.argmax(-1)


class ArabaDataHandler(SocketServer.BaseRequestHandler):

    def handle(self):
        global araba_data
        try:
            while True:
                while araba_data is not None and araba_data != "bos" and araba_data != " ":
                    self.request.send(araba_data.encode())
                    print (araba_data)
                    araba_data = "bos"
        finally:
            print ("Connection closed on thread 2")


class VideoStreamHandler(SocketServer.StreamRequestHandler):

    # create neural network
    model = NeuralNetwork()

    def handle(self):

        global araba_data
        stream_bytes = ' '
        ali = 0
        deger = 0
        solVeriSayisi = 0
        sagVeriSayisi = 0
        ileriVeriSayisi = 0

        # stream video frames one by one
        try:
            while True:
                if deger == 0:
                    #time.sleep(20)
                    deger = 1
                    baslama = cv2.getTickCount()

                #time.sleep(0.3)

                stream_bytes += self.rfile.read(1024)
                first = stream_bytes.find('\xff\xd8')
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last+2]
                    stream_bytes = stream_bytes[last+2:]
                    gray = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                    # lower half of the image
                    half_gray = image[120:240, :]

                    cv2.imshow('image', image)
                    #cv2.imshow('mlp_image', half_gray)

                    # reshape image
                    image_array = half_gray.reshape(1, 115200).astype(np.float32)

                    # neural network makes prediction
                    prediction = self.model.predict(np.float32(image_array))

                    if ali == 0:
                        print("Gaz")
                        araba_data = "G"
                        ali = 1
                    elif prediction == 2:
                        ileriVeriSayisi = ileriVeriSayisi + 1
                        sagVeriSayisi = 0
                        solVeriSayisi = 0
                        if ileriVeriSayisi % 5 == 2:
                            print("Forward")
                            araba_data = "W"
                    elif prediction == 0:
                        solVeriSayisi = solVeriSayisi + 1
                        sagVeriSayisi = 0
                        ileriVeriSayisi = 0
                        if solVeriSayisi % 15 == 8:
                            print("Left")
                            araba_data = "A"
                    elif prediction == 1:
                        sagVeriSayisi = sagVeriSayisi + 1
                        ileriVeriSayisi = 0
                        solVeriSayisi = 0
                        if sagVeriSayisi % 15 == 8:
                            print("Right")
                            araba_data = "D"
                    else :
                        print("ali")
                    '''
                    else:
                        print("Stop")
                        araba_data = "S"
                    '''

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        araba_data = "S"
                        break

            cv2.destroyAllWindows()

        finally:
            print ("Connection closed on thread 1")


class ThreadServer(object):

    def server_thread(host, port):
        server = SocketServer.TCPServer((host, port), VideoStreamHandler)
        server.serve_forever()

    def server_thread2(host, port):
        server = SocketServer.TCPServer((host, port), ArabaDataHandler)
        server.serve_forever()

    araba_thread = threading.Thread(target=server_thread2, args=('172.20.19.221', 3012))
    araba_thread.start()
    video_thread = threading.Thread(target=server_thread('172.20.19.221', 3013))
    video_thread.start()

if __name__ == '__main__':
    ThreadServer()
