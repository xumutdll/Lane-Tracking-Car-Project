import socket
import cv2
import numpy as np
import base64
import AdvancedMain as Advanced

BUFF_SIZE = 65536
PORT_FRAME = 5050
PORT_DATA = 5051
SERVER = '169.254.184.161'
ADDR_FRAME = (SERVER, PORT_FRAME)
ADDR_DATA = (SERVER, PORT_DATA)
DISCONNECT_MSG = "DISCONN"


client_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
host_name = socket.gethostname()

message = b'Hello' 
client_socket.sendto(message, ADDR_FRAME)

speed = 0
i = 0

while True:
	packet, _ = client_socket.recvfrom(BUFF_SIZE)
	data = base64.b64decode(packet,' /')
	npdata = np.frombuffer(data, dtype = np.uint8)
	frame = cv2.imdecode(npdata,1)

	speed_A, speed_B = Advanced.main(frame, speed) 
	data = str(speed_A) + " " + str(speed_B)
	client_socket.sendto(data.encode("utf-8"), ADDR_DATA)

	key = cv2.waitKey(1) &  0xFF


	if speed != 0 and i < 10:
		i += 1
	elif i == 10:
		speed = 600

	if key == ord('s'):
		speed = 700
	elif key == ord('q'):
		client_socket.sendto(DISCONNECT_MSG.encode("utf-8"), ADDR_DATA)
		client_socket.close()
		cv2.destroyAllWindows()
		break

