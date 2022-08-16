import Adafruit_PCA9685
import RPi.GPIO as GPIO

pwm = Adafruit_PCA9685.PCA9685()

pwm.set_pwm_freq(60)
pwm.set_pwm(15, 0, 0)
GPIO.setmode(GPIO.BCM) # GPIO number  in BCM mode
GPIO.setwarnings(False)


IN1 = 23  #Left motor direction pin
IN2 = 24  #Left motor dire ction pin
IN3 = 27  #Right motor direction pin
IN4 = 22  #Right motor direction pin
ENA = 0  #Left motor speed PCA9685 port 0
ENB = 1  #Right motor speed PCA9685 port 1

# Define motor control pins as output
GPIO.setup(IN1, GPIO.OUT)   
GPIO.setup(IN2, GPIO.OUT) 
GPIO.setup(IN3, GPIO.OUT)   
GPIO.setup(IN4, GPIO.OUT) 

def forward(speed_A, speed_B):
	GPIO.output(IN2, GPIO.HIGH)
	GPIO.output(IN1, GPIO.LOW)
	GPIO.output(IN4, GPIO.HIGH)
	GPIO.output(IN3, GPIO.LOW)

	pwm.set_pwm(ENA, 0, speed_A)
	pwm.set_pwm(ENB, 0, speed_B)

def stopcar():
	GPIO.output(IN1, GPIO.LOW)
	GPIO.output(IN2, GPIO.LOW)
	GPIO.output(IN3, GPIO.LOW)
	GPIO.output(IN4, GPIO.LOW)
	pwm.set_pwm(ENA, 0, 0)
	pwm.set_pwm(ENB, 0, 0)