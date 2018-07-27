import threading
import SocketServer
import numpy as np
import cv2
import pygame
from pygame.locals import *
import time
import os

araba_data = " "

class ArabaDataHandler(SocketServer.BaseRequestHandler):

    def handle(self):
        global araba_data
        try:
            while True:
                while araba_data is not None and araba_data != "bos" and araba_data != " ":
                    self.request.send(araba_data.encode())
                    print araba_data
                    araba_data = "bos"
        finally:
            print "Connection closed on thread 2"

class VideoStreamHandler(SocketServer.StreamRequestHandler):

    def handle(self):
        global araba_data

        # create labels
        self.k = np.zeros((4, 4), 'float')
        for i in range(4):
            self.k[i, i] = 1
        self.temp_label = np.zeros((1, 4), 'float')

        pygame.init()

        saved_frame = 0
        total_frame = 0

        # collect images for training
        print('Start collecting images...')
        e1 = cv2.getTickCount()
        image_array = np.zeros((1, 115200))
        label_array = np.zeros((1, 4), 'float')

        stream_bytes = ''
        frame = 1

        self.send_inst = True

        try:
            while True:
                print('Start collecting images...')
                e1 = cv2.getTickCount()
                frame = 1
                saved_frame = 0
                total_frame = 0
                basarili = True
                image_array = np.zeros((1, 115200))
                label_array = np.zeros((1, 4), 'float')
                while self.send_inst:
                    stream_bytes += self.rfile.read(1024)
                    first = stream_bytes.find('\xff\xd8')
                    last = stream_bytes.find('\xff\xd9')

                    if first != -1 and last != -1:
                        jpg = stream_bytes[first:last + 2]
                        stream_bytes = stream_bytes[last + 2:]
                        image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                        # select lower half of the image
                        roi = image[120:240, :]

                        # save streamed images
                        cv2.imwrite('training_images/frame{:>05}.jpg'.format(frame), image)

                        #cv2.imshow('roi_image', roi)
                        cv2.imshow('image', image)

                        # reshape the roi image into one row array
                        #image_array = np.zeros(roi.shape)
                        temp_array = roi.reshape(1, 115200).astype(np.float32)

                        frame += 1
                        total_frame += 1

                        for event in pygame.event.get():
                            if event.type == KEYDOWN:
                                key_input = pygame.key.get_pressed()

                                if key_input[pygame.K_SPACE]:
                                    print("Gaz")
                                    #self.ser.write(chr(1)
                                    araba_data = "G"

                                elif key_input[pygame.K_UP]:
                                    print("Forward")
                                    saved_frame += 1
                                    image_array = np.vstack((image_array, temp_array))
                                    label_array = np.vstack((label_array, self.k[2]))
                                    #self.ser.write(chr(1)
                                    araba_data = "W"

                                elif key_input[pygame.K_DOWN]:
                                    print("Stop")
                                    #self.ser.write(chr(1)
                                    araba_data = "S"

                                elif key_input[pygame.K_RIGHT]:
                                    print("Right")
                                    image_array = np.vstack((image_array, temp_array))
                                    label_array = np.vstack((label_array, self.k[1]))
                                    saved_frame += 1
                                    #self.ser.write(chr(3))
                                    araba_data = "D"

                                elif key_input[pygame.K_LEFT]:
                                    print("Left")
                                    image_array = np.vstack((image_array, temp_array))
                                    label_array = np.vstack((label_array, self.k[0]))
                                    saved_frame += 1
                                    #self.ser.write(chr(4))
                                    araba_data = "A"

                                elif key_input[pygame.K_o]:
                                    print ('servo kapat')
                                    araba_data = "O"
                                    #self.ser.write(chr(0))

                                elif key_input[pygame.K_q]:
                                    print ('basarili exit')
                                    self.send_inst = False
                                    #self.ser.write(chr(0))
                                    break

                                elif key_input[pygame.K_x]:
                                    print ('basarisiz exit')
                                    self.send_inst = False
                                    basarili = False
                                    break

                if basarili:
                    # save training images and labels
                    train = image_array[1:, :]
                    train_labels = label_array[1:, :]

                    # save training data as a numpy file
                    file_name = str(int(time.time()))
                    directory = "training_data"
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    try:
                        np.savez(directory + '/' + file_name + '.npz', train=train, train_labels=train_labels)
                    except IOError as e:
                        print(e)

                    e2 = cv2.getTickCount()
                    # calculate streaming duration
                    time0 = (e2 - e1) / cv2.getTickFrequency()
                    print ('Streaming duration:', time0)

                    print(train.shape)
                    print(train_labels.shape)
                    print ('Total frame:', total_frame)
                    print ('Saved frame:', saved_frame)
                    print ('Dropped frame', total_frame - saved_frame)

                self.send_inst = True

        finally:
            self.rfile.close()
            print "Connection closed on thread 1"

class ThreadServer(object):

    def server_thread(host, port):
        server = SocketServer.TCPServer((host, port), VideoStreamHandler)
        server.serve_forever()

    def server_thread2(host, port):
        server = SocketServer.TCPServer((host, port), ArabaDataHandler)
        server.serve_forever()

    araba_thread = threading.Thread(target=server_thread2, args=('172.20.16.214', 3068))
    araba_thread.start()
    video_thread = threading.Thread(target=server_thread('172.20.16.214', 3069))
    video_thread.start()

if __name__ == '__main__':
    ThreadServer()
