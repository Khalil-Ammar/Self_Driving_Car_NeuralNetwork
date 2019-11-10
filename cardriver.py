from gpiozero import PWMOutputDevice
from gpiozero import DigitalOutputDevice

###CAR SETUP
PWM_THRUST_PIN = 13
THRUST_FWD_PIN = 24
THRUST_BWD_PIN = 23

PWM_TURN_PIN = 19
TURN_LEFT_PIN = 10
TURN_RIGHT_PIN = 9

pwmThrust = PWMOutputDevice(PWM_THRUST_PIN, True, 0, 1000)
pwmTurn = PWMOutputDevice(PWM_TURN_PIN, True, 0, 1000)

pwmTurn.value = 1.0

thrustFwd = DigitalOutputDevice(THRUST_FWD_PIN)
thrustBwd = DigitalOutputDevice(THRUST_BWD_PIN)
turnLeft = DigitalOutputDevice(TURN_LEFT_PIN)
turnRight = DigitalOutputDevice(TURN_RIGHT_PIN)
THRUSTVAL = 0.5
def forward():
	thrustFwd.value = True
	thrustBwd.value = False
	turnLeft.value = False
	turnRight.value = False
	pwmThrust.value = THRUSTVAL
	pwmTurn.value = 1.0

def backward():
	thrustFwd.value = False
	thrustBwd.value = True
	turnLeft.value = False
	turnRight.value = False
	pwmThrust.value = THRUSTVAL
	pwmTurn.value = 1.0

def left():

	thrustFwd.value = True
	thrustBwd.value = False
	turnLeft.value = True
	turnRight.value = False
	pwmThrust.value = THRUSTVAL
	pwmTurn.value = 1.0

def right():

	thrustFwd.value = True
	thrustBwd.value = False
	turnLeft.value = False
	turnRight.value = True
	pwmThrust.value = THRUSTVAL
	pwmTurn.value = 1.0

def stop():
	thrustFwd.value = False
	thrustBwd.value = False
	turnLeft.value = False
	turnRight.value = False
	pwmThrust.value = 0
	pwmTurn.value = 0
