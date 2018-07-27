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


class ObjectDetection(object):

    def detect(self, cascade_classifier, gray_image, image):

        stop = False

        # y camera coordinate of the target point 'P'
        v = 0

        # detection
        cascade_obj = cascade_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # draw a rectangle around the objects
        for (x_pos, y_pos, width, height) in cascade_obj:
            cv2.rectangle(image, (x_pos+5, y_pos+5), (x_pos+width-5, y_pos+height-5), (255, 255, 255), 2)
            v = y_pos + height - 5

            # stop sign
            if width/height == 1:
                cv2.putText(image, 'STOP', (x_pos, y_pos-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print("Stop goruldu")
                stop = True


        return stop


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

    obj_detection = ObjectDetection()

    # cascade classifiers
    stop_cascade = cv2.CascadeClassifier('cascade_xml/stop_sign.xml')

    stop_start = 0              # start time when stop at the stop sign
    stop_finish = 0
    stop_time = 0
    drive_time_after_stop = 0

    def handle(self):

        global araba_data
        stream_bytes = ' '
        ali = 0
        deger = 0
        solVeriSayisi = 0
        sagVeriSayisi = 0
        ileriVeriSayisi = 0
        stop_flag = False
        stop_sign_active = True

        # stream video frames one by one
        try:
            while True:

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

                    # object detection
                    v_param1 = self.obj_detection.detect(self.stop_cascade, gray, image)

                    cv2.imshow('image', image)
                    #cv2.imshow('mlp_image', half_gray)

                    # reshape image
                    image_array = half_gray.reshape(1, 115200).astype(np.float32)

                    # neural network makes prediction
                    prediction = self.model.predict(np.float32(image_array))

                    if v_param1 and stop_sign_active:
                        print("Stop sign ahead")
                        print("Dur")
                        #araba_data = "S"

                        # stop for 5 seconds
                        if stop_flag is False:
                            self.stop_start = cv2.getTickCount()
                            stop_flag = True
                        self.stop_finish = cv2.getTickCount()

                        self.stop_time = (self.stop_finish - self.stop_start)/cv2.getTickFrequency()
                        print "Stop time: %.2fs" % self.stop_time

                        # 5 seconds later, continue driving
                        if self.stop_time > 5:
                            print("Waited for 5 seconds")
                            stop_flag = False
                            stop_sign_active = False
                            araba_data = "G"

                    else:
                        if ali == 0:
                            print("Gaz")
                            araba_data = "G"
                            ali = 1
                        elif prediction == 2:
                            ileriVeriSayisi = ileriVeriSayisi + 1
                            sagVeriSayisi = 0
                            solVeriSayisi = 0
                            if ileriVeriSayisi % 7 == 3:
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

                        self.stop_start = cv2.getTickCount()

                        if stop_sign_active is False:
                            self.drive_time_after_stop = (self.stop_start - self.stop_finish)/cv2.getTickFrequency()
                            if self.drive_time_after_stop > 5:
                                stop_sign_active = True

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

    araba_thread = threading.Thread(target=server_thread2, args=('172.20.16.145', 3002))
    araba_thread.start()
    video_thread = threading.Thread(target=server_thread('172.20.16.145', 3003))
    video_thread.start()

if __name__ == '__main__':
    ThreadServer()
