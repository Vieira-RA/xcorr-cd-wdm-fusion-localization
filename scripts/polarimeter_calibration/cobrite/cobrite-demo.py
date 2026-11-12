
import socket


class CoBrite:
    def __init__(self, host='cobrite.local', port=2000):
        self.host = host
        self.port = port
        self.socket = socket.socket()
        self.socket.connect((self.host, self.port))

    def sendScpi(self, command):
        print("TX: " + command)
        self.socket.sendall(bytearray(command + "\n", 'utf-8'))
        reply = ""
        while reply.find('\n') < 0:
            reply = reply + self.socket.recv(1024).decode("utf-8")
        print("RX: " + reply)
        return reply

    def close():
        self.socket.close()


cobrite = CoBrite()
cobrite.sendScpi('*idn?')
cobrite.sendScpi('freq 192.0')
cobrite.sendScpi('*wai')
cobrite.sendScpi('freq?')
