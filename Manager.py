import numpy as np

import Communicator
from socket import socket, AF_INET, SOCK_DGRAM
import threading


class CommunicationManager:
    def __init__(self, port, address, botcount):
        self.port = port
        self.address = address
        self.results = None
        self.running = False
        self.botcount = botcount
        self.changedbotcount = False
        self.shuffle = False
        self.eog = False
        self.start = False
        self.resultsarethere = False
        self.sender = threading.Thread(target=self.sender())
        self.receiver = threading.Thread(target=self.reveiver())
        self.botlist = []

    def setbotcount(self, count):
        self.botcount = count
        self.changedbotcount = True

    def checkresults(self):
        if self.resultsarethere:
            self.resultsarethere = False
            return True
        return False

    def setstart(self):
        self.start = True

    def setstop(self):
        self.eog = True

    def setshuffle(self):
        self.shuffle = True

    def getresults(self):
        return self.results

    def startcommunication(self):
        self.sender.start()
        self.receiver.start()

    def reciver(self):
        sock = socket(AF_INET, SOCK_DGRAM)
        sock.bind((self.address, self.port + 1))
        results = ""
        resultsevent = False
        while True:
            if self.running:
                break
            msg, addr = sock.recvfrom(8654)  # This is the amount of bytes to read at maximum
            if msg == bytes("RESULTSTART", 'utf-8'):
                resultsevent = True
                results = ""
                continue
            if msg == bytes("RESULTEND", 'utf-8'):
                resultsevent = False
                results = results.split(";")
                self.results = np.array(results).astype(np.float)
                self.resultsarethere = True
                results = ""
                continue
            if resultsevent:
                results += msg.decode('utf-8')

    def sender(self):
        sock = socket(AF_INET, SOCK_DGRAM)
        sock.bind((self.address, self.port + 3))
        while True:
            if self.running:
                break
            if self.shuffle:
                sock.sendto(bytes("SHUFFLE", "utf-8"), (self.address, self.port + 2))
            if self.start:
                sock.sendto(bytes("START", "utf-8"), (self.address, self.port + 2))
            if self.changedbotcount:
                sock.sendto(bytes(str(self.botcount), "utf-8"), (self.address, self.port + 2))
            if self.eog:
                sock.sendto(bytes("EOG", "utf-8"), (self.address, self.port + 2))
