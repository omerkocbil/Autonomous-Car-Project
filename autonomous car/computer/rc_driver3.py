__author__ = 'omer'

import threading
import SocketServer
import cv2
import numpy as np
import math
from sklearn.model_selection import train_test_split
import sys
import glob

# distance data measured by ultrasonic sensor
araba_data = " "


class NeuralNetwork(object):

    def __init__(self):
        self.model = cv2.ml.ANN_MLP_load('mlp_xml/mlp.xml')

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
    #model = NeuralNetwork()
    model = cv2.ml.ANN_MLP_create()

    ali = 0

    def handle(self):

        global araba_data
        stream_bytes = ' '
        ali = 0
        deger = 0

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
                    print(half_gray.shape)

                    cv2.imshow('image', image)
                    #cv2.imshow('mlp_image', half_gray)

                    # reshape image
                    image_array = half_gray.reshape(1, 115200).astype(np.float32)

                    if ali == 0:
                        ali = 1
                        print ('Loading training data...')
                        e0 = cv2.getTickCount()

                        # load training data
                        image_array2 = np.zeros((1, 115200))
                        #image_array = np.zeros((119, 320, 3))
                        label_array2 = np.zeros((1, 4), 'float')
                        training_data = glob.glob('training_data/*.npz')

                        # if no data, exit
                        if not training_data:
                            print ("No training data in directory, exit")
                            sys.exit()

                        for single_npz in training_data:
                            with np.load(single_npz) as data:
                                train_temp2 = data['train']
                                train_labels_temp2 = data['train_labels']
                            image_array2 = np.vstack((image_array2, train_temp2))
                            label_array2 = np.vstack((label_array2, train_labels_temp2))

                        X = image_array2[1:, :]
                        y = label_array2[1:, :]
                        print ('Image array shape: ', X.shape)
                        print ('Label array shape: ', y.shape)

                        e00 = cv2.getTickCount()
                        time0 = (e00 - e0)/ cv2.getTickFrequency()
                        print ('Loading image duration:', time0)

                        # train test split, 7:3
                        train, test, train_labels, test_labels = train_test_split(X, y, test_size=0.3)

                        # set start time
                        setStart = cv2.getTickCount()

                        # create MLP
                        layer_sizes = np.int32([115200, 32, 4])

                        #model.create(layer_sizes)

                        self.model.setLayerSizes(layer_sizes)
                        self.model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
                        self.model.setBackpropMomentumScale(0.0)
                        self.model.setBackpropWeightScale(0.001)
                        self.model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 20, 0.01))
                        self.model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)

                        #criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.0001)
                        #criteria2 = (cv2.TERM_CRITERIA_COUNT, 100, 0.001)
                        #params = dict(term_crit = criteria,
                                       #trasin_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,
                                       #bp_dw_scale = 0.001,
                                       #bp_moment_scale = 0.0 )

                        print ('Training MLP ...')
                        num_iter = self.model.train(np.float32(train), cv2.ml.ROW_SAMPLE, np.float32(train_labels))

                        # set end time
                        setFinish = cv2.getTickCount()
                        time = (setFinish - setStart)/cv2.getTickFrequency()
                        print ('Training duration:', time)
                        #print 'Ran for %d iterations' % num_iter

                        # train data
                        ret_0, resp_0 = self.model.predict(train)
                        prediction_0 = resp_0.argmax(-1)
                        true_labels_0 = train_labels.argmax(-1)

                        train_rate = np.mean(prediction_0 == true_labels_0)
                        print ('Train accuracy: ', "{0:.2f}%".format(train_rate * 100))

                        # test data
                        ret_1, resp_1 = self.model.predict(test)
                        prediction_1 = resp_1.argmax(-1)
                        true_labels_1 = test_labels.argmax(-1)

                        test_rate = np.mean(prediction_1 == true_labels_1)
                        print ('Test accuracy: ', "{0:.2f}%".format(test_rate * 100))

                        ret, resp = self.model.predict(np.float32(image_array))
                        prediction = resp.argmax(-1)
                        print(prediction)
                        print("Aliiiiiii")

                    # neural network makes prediction
                    #prediction = self.model.predict(image_array)
                    #ret, resp = self.model.predict(np.float32(image_array))
                    #prediction = resp.argmax(-1)


                    if ali == 0:
                        print("Gaz")
                        araba_data = "G"
                        ali = 1

                    if prediction == 2:
                        print("Forward")
                        araba_data = "W"
                    elif prediction == 0:
                        print("Left")
                        araba_data = "A"
                    elif prediction == 1:
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

    araba_thread = threading.Thread(target=server_thread2, args=('172.20.16.71', 4572))
    araba_thread.start()
    video_thread = threading.Thread(target=server_thread('172.20.16.71', 4573))
    video_thread.start()

if __name__ == '__main__':
    ThreadServer()
