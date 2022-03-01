import sys
import socket
import numpy as np
import cv2

import NeuralNetwork


class communicator:
    def __init__(self, port: int, recport: int, neuralnet: NeuralNetwork):
        self.port = port
        self.neuralnet = neuralnet
        self.socket = socket.socket(socket.AF_INET,  # Internet
                                    socket.SOCK_DGRAM)  # UDP
        self.socket.bind(("", self.port))

    def sendwelcome(self, address):
        startstatement = "Begin"
        self.socket.sendto(startstatement, address)
        while True:
            try:
                data, addr = self.socket.recvfrom(1024)
                if str(data) == startstatement:
                    break
            except socket.error:
                pass
        while True:
            try:
                data, addr = self.socket.recvfrom(1024)
                if int(data) == 0:
                    break
                data, addr = self.socket.recvfrom(int(data))
                steering = self.neuralnet.neuralnet(self.imagetoposition(data))
                self.socket.sendto(steering[0],address)
                self.socket.sendto(steering[1], address)
            except socket.error:
                pass

    def imagetoposition(self, image: bytes):

        img = cv2.imread(np.asarray(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        height, width, _ = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # (hMin = 3, sMin = 106, vMin = 0), (hMax = 60, sMax = 202, vMax = 165)
        lower_band = np.array([3, 106, 0])
        upper_bande = np.array([60, 202, 165])
        bande = cv2.inRange(hsv, lower_band, upper_bande)
        ret, bande_img = cv2.threshold(bande, 70, 255, cv2.THRESH_BINARY)
        # (hMin = 0, sMin = 0, vMin = 125), (hMax = 179, sMax = 242, vMax = 131)
        lower_red = np.array([0, 0, 125])
        upper_red = np.array([179, 242, 131])
        red = cv2.inRange(hsv, lower_red, upper_red)
        ret, red_img = cv2.threshold(red, 70, 255, cv2.THRESH_BINARY)
        # (hMin = 0, sMin = 88, vMin = 0), (hMax = 179, sMax = 255, vMax = 255) all execpt backround lower_all =
        # np.array([0, 88, 0]) upper_all = np.array([179, 255, 255]) (hMin = 0, sMin = 124, vMin = 73), (hMax = 179,
        # sMax = 244, vMax = 94) only blue (hMin = 91 , sMin = 102, vMin = 0), (hMax = 171 , sMax = 255, vMax = 255)
        upper_blue = np.array([179, 244, 94], np.uint8)
        lower_blue = np.array([0, 124, 73], np.uint8)
        blue = cv2.inRange(hsv, lower_blue, upper_blue)
        ret, blue_img = cv2.threshold(blue, 70, 255, cv2.THRESH_BINARY)
        contours_blue, _ = cv2.findContours(blue_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(red_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_bande, _ = cv2.findContours(bande_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        redreq = []
        bluereg = []
        bandreg = []
        for cnt in contours_red:
            # Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if area > 10:
                (x, y, w, h) = cv2.boundingRect(cnt)
                bound = np.asarray([x, y, w, h])
                redreq.append(bound)

        for cnt in contours_blue:
            area = cv2.contourArea(cnt)
            if area > 10:
                (x, y, w, h) = cv2.boundingRect(cnt)
                bound = np.asarray([x, y, w, h])
                bluereg.append(bound)

        for cnt in contours_bande:
            area = cv2.contourArea(cnt)
            if area > 10:
                (x, y, w, h) = cv2.boundingRect(cnt)
                bound = np.asarray([x, y, w, h])
                bandreg.append(bound)

        return [redreq, bluereg, bandreg]
