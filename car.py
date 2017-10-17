

try:
    import RPi.GPIO as GPIO
    import PCA9685 as servo
except ImportError as e:
    print("error importing hardware control library, running on the car?")
    print(e)

    class Car:
        def steering(self, v):
            angle = 450 + 50 * v
            print("car: steering, angle %f" % angle)

        def throttle(self, v):
            print("car: throttle %f" % v)

else:

    Motor0_A = 11  # pin11
    Motor0_B = 12  # pin12
    Motor1_A = 13  # pin13
    Motor1_B = 15  # pin15
    EN_M0 = 4  # servo driver IC CH4
    EN_M1 = 5  # servo driver IC CH5
    pins = [Motor0_A, Motor0_B, Motor1_A, Motor1_B]

    class Car:

        def __init__(self):
            self.pwm = servo.PWM()
            self.pwm.write(0, 0, 0)
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BOARD)
            for pin in pins:
                GPIO.setup(pin, GPIO.OUT)
            self._forward()

        def _forward(self):
            GPIO.output(Motor0_A, GPIO.LOW)
            GPIO.output(Motor0_B, GPIO.HIGH)
            GPIO.output(Motor1_A, GPIO.LOW)
            GPIO.output(Motor1_B, GPIO.HIGH)

        def _backward(self):
            GPIO.output(Motor0_A, GPIO.HIGH)
            GPIO.output(Motor0_B, GPIO.LOW)
            GPIO.output(Motor1_A, GPIO.HIGH)
            GPIO.output(Motor1_B, GPIO.LOW)


        def steering(self, v):
            angle = int(420 + 150 * v)
            self.pwm.write(0, 0, angle)

        def throttle(self, v):
            if v < 0:
              self._backward()
            else:
              self._forward()
            speed = int(abs(v) * 8000)
            self.pwm.write(EN_M0, 0, speed)
            self.pwm.write(EN_M1, 0, speed)

        def stop(self):
            for pin in pins:
                GPIO.output(pin, GPIO.LOW)
