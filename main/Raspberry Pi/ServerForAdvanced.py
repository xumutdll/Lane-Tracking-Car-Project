import socket
import cv2
import numpy as np
import base64
import concurrent.futures
import Control

BUFF_SIZE = 65536
PORT_FRAME = 5050
PORT_DATA = 5051
SERVER = "169.254.184.161"
ADDR_FRAME = (SERVER, PORT_FRAME)
ADDR2_DATA = (SERVER, PORT_DATA)
DISCONNECT_MSG = "DISCONN"

server_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_sender.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
server_sender.bind(ADDR_FRAME)

server_receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16)
server_receiver.bind(ADDR2_DATA)



print('Waiting for connection..')
msg,client_addr = server_sender.recvfrom(BUFF_SIZE)
print('GOT connection from ',client_addr)


camera = cv2.VideoCapture(-1)
camera.set(3, 640)
camera.set(4, 480)


def data_reciving():
	while True:
		data, _ = server_receiver.recvfrom(16)
		data = data.decode("utf-8")

		if data == DISCONNECT_MSG:
			Control.stopcar()
			camera.release()
			server_sender.close()
			server_receiver.close()
			break
		
		speed_A, speed_B = data.split()
		Control.forward(int(speed_A), int(speed_B))
		print(data)


def camera_sending():
	while (camera.isOpened()):
		_,frame = camera.read()
		buffer, encoded = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
		img = base64.b64encode(encoded)
		server_sender.sendto(img, client_addr)

		
with concurrent.futures.ProcessPoolExecutor() as executor:
	executor.submit(data_reciving)
	executor.submit(camera_sending)
	