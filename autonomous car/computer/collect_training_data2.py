__author__ = 'omer'

import numpy as np
import cv2
import pygame
from pygame.locals import *
import socket
import time
import os
import select
import string


class CollectTrainingData(object):

    def __init__(self):
        '''
        self.server_socket = socket.socket()
        self.server_socket.bind(('172.16.17.178', 8045))
        self.server_socket.listen(0)

        # accept a single connection
        self.connection = self.server_socket.accept()[0].makefile('rb')
        '''

        # connect to a seral port
        self.send_inst = True

        # create labels
        self.k = np.zeros((4, 4), 'float')
        for i in range(4):
            self.k[i, i] = 1
        self.temp_label = np.zeros((1, 4), 'float')

        pygame.init()

        # List to keep track of socket descriptors
        self.CONNECTION_LIST=[]

        # Do basic steps for server like create, bind and listening on the socket

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("172.16.18.8", 9019))
        self.server_socket.listen(10)

        # Add server socket to the list of readable connections
        self.CONNECTION_LIST.append(self.server_socket)

        print ("TCP/IP Chat server process started.")

        saved_frame = 0
        total_frame = 0

        # collect images for training
        print('Start collecting images...')
        e1 = cv2.getTickCount()
        #image_array = np.zeros((1, 38400))
        label_array = np.zeros((1, 4), 'float')

        stream_bytes = ''
        frame = 1

        ali = 0
        self.sockfd = ''
        self.addr = ''

        while self.send_inst:
            # Get the list sockets which are ready to be read through select
            read_sockets,write_sockets,error_sockets = select.select(self.CONNECTION_LIST,[],[])
            for sock in read_sockets:
                if sock == self.server_socket:
                    # Handle the case in which there is a new connection recieved
                    # through server_socket
                    if ali == 0:
                        self.sockfd = self.server_socket.accept()[0].makefile('rb')
                        ali = ali + 1
                    elif ali == 1:
                        self.sockfd = self.server_socket.accept()[0].makefile('wb')
                        ali = ali + 1

                    self.CONNECTION_LIST.append(self.sockfd)
                    print ("Client connected")
                    self.broadcast_data(self.sockfd, "Client connected")

                else:
                    # Data recieved from client, process it
                    try:
                        #In Windows, sometimes when a TCP program closes abruptly,
                        # a "Connection reset by peer" exception will be thrown

                        stream_bytes += self.sockfd.read(1024)
                        #stream_bytes = sock.recv(4096)
                        first = stream_bytes.find('\xff\xd8')
                        last = stream_bytes.find('\xff\xd9')
                    except:
                        self.broadcast_data(sock, "Client is offline")
                        print ("Client is offline")
                        sock.close()
                        self.CONNECTION_LIST.remove(sock)
                        continue

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
                        image_array = np.zeros(roi.shape)
                        temp_array = roi.astype(np.float32)

                        frame += 1
                        total_frame += 1

                        for event in pygame.event.get():
                            if event.type == KEYDOWN:
                                key_input = pygame.key.get_pressed()

                                if key_input[pygame.K_UP]:
                                    print("Forward")
                                    saved_frame += 1
                                    image_array = np.vstack((image_array, temp_array))
                                    label_array = np.vstack((label_array, self.k[2]))
                                    #self.ser.write(chr(1)
                                    self.broadcast_data(sock, "A")

                                elif key_input[pygame.K_RIGHT]:
                                    print("Right")
                                    image_array = np.vstack((image_array, temp_array))
                                    label_array = np.vstack((label_array, self.k[1]))
                                    saved_frame += 1
                                    #self.ser.write(chr(3))
                                    self.broadcast_data(sock, "A")

                                elif key_input[pygame.K_LEFT]:
                                    print("Left")
                                    image_array = np.vstack((image_array, temp_array))
                                    label_array = np.vstack((label_array, self.k[0]))
                                    saved_frame += 1
                                    #self.ser.write(chr(4))
                                    self.broadcast_data(sock, "A")

                                elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                                    print ('exit')
                                    self.send_inst = False
                                    #self.ser.write(chr(0))
                                    break

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

        self.server_socket.close()

    def broadcast_data (self, sock, message):
        """Send broadcast message to all clients other than the
           server socket and the client socket from which the data is received."""

        for socket in self.CONNECTION_LIST:
            if socket != self.server_socket and socket != sock:
                socket.write(message.encode())
                print("Aliiiiiiiiiii")

if __name__ == '__main__':
    CollectTrainingData()
