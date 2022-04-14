import sys
from socket import socket, AF_INET, SOCK_DGRAM
import numpy as np
import cv2
import threading
import NeuralNetwork


class Communicator:
    def __init__(self, port: int, address: str, neuralnet: NeuralNetwork):
        self.port = port
        self.address = address
        self.neuralnet = neuralnet
        self.running = True
        self.dataready = False
        self.datatobesend: str = ""
        sender = threading.Thread(target=self.sender())
        receiver = threading.Thread(target=self.receiver())
        sender.start()
        receiver.start()

    def receiver(self):
        sock = socket(AF_INET, SOCK_DGRAM)
        sock.bind((self.address, self.port + 3))
        pic = []
        while True:
            if self.running:
                break
            msg, addr = sock.recvfrom(8654)  # This is the amount of bytes to read at maximum
            if msg == bytes("LosGehtsKleinerHase", 'utf-8'):
                pic = []
                continue
            if msg == bytes("ZuEndekleinerHase", 'utf-8'):
                print(self.imagetoposition(bytearray(pic)))
                continue
            pic += msg

    def sender(self):
        sock = socket(AF_INET, SOCK_DGRAM)
        sock.bind((self.address, self.port + 1))
        while True:
            if self.running:
                break
            if self.dataready:
                sock.sendto(bytes(self.datatobesend, "utf-8"), (self.address, self.port + 2))
                self.dataready = False

    def imagetoposition(self, image: bytes):
        img = cv2.imdecode(np.asarray(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        height, width, _ = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_band = np.array([3, 106, 0])
        upper_bande = np.array([60, 202, 165])
        bande = cv2.inRange(hsv, lower_band, upper_bande)
        ret, bande_img = cv2.threshold(bande, 70, 255, cv2.THRESH_BINARY)
        lower_red = np.array([0, 0, 125])
        upper_red = np.array([179, 242, 131])
        red = cv2.inRange(hsv, lower_red, upper_red)
        ret, red_img = cv2.threshold(red, 70, 255, cv2.THRESH_BINARY)
        upper_blue = np.array([114, 231, 138], np.uint8)
        lower_blue = np.array([110, 0, 99], np.uint8)
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
