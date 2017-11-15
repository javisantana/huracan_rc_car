

try:
    import RPi.GPIO as GPIO
    import PCA9685 as servo
    import cv2
    import numpy as np
    import matplotlib
    from StringIO import StringIO
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError as e:
    import os
    print("error importing hardware control library, running on the car?")
    print(e)

    # this is a dummy implementation for the car
    class Car:
        def __init__(self):
            self.camera = FakeCamera()

        def steering(self, v):
            angle = 450 + 50 * v
            print("car: steering, angle %f" % angle)

        def throttle(self, v):
            print("car: throttle %f" % v)

        def start_record_images(self, folder, interval):
            print("start recording images")

        def stop_record_images(self):
            print("finish recording images")

        # im = ndimage.imread('../records/record_Wed_18_Oct_2017-18_23_26/20.jpg', mode='L')

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
            self.camera = Camera()

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


#
# Camera module provides with the basics to get the image from the camera and
# record it in the disk (so later can be reproduced)
# It uses opencv to capute data from the camera so a thread loop reads the images
#
import threading
import time
from sensor_camera import extract_lines
from scipy import ndimage
class Camera:
    def __init__(self, capture_interval=0.5):
        self.recording = False
        self.cap = cv2.VideoCapture(0)
        self.capture_interval = capture_interval
        self.last_image = None

    def start(self, folder=None, capture_interval=0.5):
        """ starts recording images, if `folder` is set saves images in that folder """
        self.recording = True
        self.capture_interval = capture_interval
        def _record():
            frame = 0
            while self.recording:
                ret, im = self.cap.read()
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                im = ndimage.zoom(im, 0.1)
                t0 = time.time()
                extract_lines(im, plt)
                t1 = time.time()
                print ("processing time %f" % (t1 - t0))
                if ret and folder:
                    cv2.imwrite(folder + '/%04d.jpg' % frame, im)
                    frame += 1
                    time.sleep(self.capture_interval)
                else:
                    print("error capturing image")

                output = StringIO()
                plt.savefig(output)
                #encoded_string = base64.b64encode(output.getvalue())
                self.last_image = output.getvalue() #im
        self.record_image_thread = threading.Thread(target=_record)
        self.record_image_thread.start()

    def stop(self):
        """stops the recording"""
        self.recording = False
        self.record_image_thread.join()

    def get_last_image(self):
        """ returns the lastest captured image. None if no image was already captures """
        im = None
        if self.last_image != None:
            #ret, im = cv2.imencode('.jpg', self.last_image)
            #im = np.getbuffer(im)
            im = self.last_image
        return im
        #return bytearray(im.flatten().tolist())

class FakeCamera:
    def __init__(self, capture_interval=0.5):
        self.recording = False
        self.capture_interval = capture_interval
        self.last_image = None
        # create a generator to read images

    def start(self, folder=None, capture_interval=0.5):
        self.folder = folder
        def gen():
            images = os.listdir(self.folder)
            while 1:
                for im in images:
                    if '.jpg' in im:
                        d = open(self.folder + "/" + im)
                        data = d.read()
                        d.close()
                        yield data
        self.images = gen()
        self.recording = True
        self.capture_interval = capture_interval
        def _record():
            while self.recording:
                self.last_image = self.images.next()
                time.sleep(self.capture_interval)
        self.record_image_thread = threading.Thread(target=_record)
        self.record_image_thread.start()

    def stop(self):
        self.recording = False
        self.record_image_thread.join()

    def get_last_image(self):
        return self.last_image
        #return bytearray(im.flatten().tolist())
