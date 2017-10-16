

try:
    import PCA9685 as servo
except ImportError as e:
    print("error importing hardware control library, running on the car?")
    print (e)
    class Car:
        def steering(self, v):
            angle = 450 + 50 * v
            print("car: steering, angle %f" % angle)

else:

    class Car:

        def __init__(self):
            self.pwm = servo.PWM()
            self.pwm.write(0, 0, 0)

        def steering(self, v):
            angle = int(420 + 150 * v)
            self.pwm.write(0, 0, angle)
