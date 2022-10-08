#
"""Project using TF Lite and Raspberry Pi Camera to detect objects and report
to visually impaired user for ease of navigation"""
#
# Progress:
#
# works wTTS; separating objects into directions - left,middle,right; and calculating biggest to announce
# biggest 3 objects
#
# calculates vibration intensity of 0 to 1 and output to haptic motor
#
# only announces biggest object
#
# Added ultrasonic sensor for actual detection of proximity and removed
# previous vibration calculation by size of bounding boxes

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import re
import time

from annotation import Annotator

import numpy as np
import picamera

from PIL import Image
from tflite_runtime.interpreter import Interpreter

import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate',150)

from gpiozero import InputDevice, OutputDevice, PWMOutputDevice
#from time import sleep, time

#setup ultrasonic sensor
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)

GPIO_TRIGGER = 4
GPIO_ECHO = 17

GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

HAPTIC_MOTOR = 14

#setup haptic motor
motor = PWMOutputDevice(HAPTIC_MOTOR)

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
NUM_VOCALIZED = 1

def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels


def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))
 
  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results


def annotate_objects(annotator, results, labels):
  """Draws the bounding box and label for each object in the results."""
  for obj in results:
    # Convert the bounding box figures from relative coordinates
    # to absolute coordinates based on the original resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * CAMERA_WIDTH)
    xmax = int(xmax * CAMERA_WIDTH)
    ymin = int(ymin * CAMERA_HEIGHT)
    ymax = int(ymax * CAMERA_HEIGHT)

    # Overlay the box, label, and score on the camera preview
    annotator.bounding_box([xmin, ymin, xmax, ymax])
    annotator.text([xmin, ymin],
                   '%s\n%.2f' % (labels[obj['class_id']], obj['score']))

def vocalize_objects(results, labels):
   
    if (len(results)<NUM_VOCALIZED):
        for obj in results:
            ymin, xmin, ymax, xmax = obj['bounding_box']
            xmax = int(xmax * CAMERA_WIDTH)
            xmin = int(xmin * CAMERA_WIDTH)
            xmid = int((xmax+xmin)/2)
            if (xmid<213):
                location = 'left'
            elif (xmid>213 and xmid<427):
                location = 'middle'
            else:
                location = 'right'    
            engine.say(labels[obj['class_id']]+ location)
            engine.runAndWait()
    else:
        for i in range(NUM_VOCALIZED):
            obj = results[i]
            ymin, xmin, ymax, xmax = obj['bounding_box']
            xmax = int(xmax * CAMERA_WIDTH)
            xmin = int(xmin * CAMERA_WIDTH)
            xmid = int((xmax+xmin)/2)
            if (xmid<213):
                location = 'left'
            elif (xmid>213 and xmid<427):
                location = 'middle'
            else:
                location = 'right'    
            engine.say(labels[obj['class_id']]+ location)
            engine.runAndWait()
            #vibration = calculate_vibration(obj)
            #print(vibration)
            #motor.value = vibration
            #time.sleep(0.25)

def sorting_objects(results, labels):
    n = len(results)
    #print('before sort:')
    for obj in results:
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * CAMERA_WIDTH)
        xmax = int(xmax * CAMERA_WIDTH)
        ymin = int(ymin * CAMERA_HEIGHT)
        ymax = int(ymax * CAMERA_HEIGHT)
        width = xmax-xmin
        height = ymax-ymin
        area1 = width*height
        #print(labels[obj['class_id']], '/', area1, end=' ')
    #print('')
   
    for i in range(0,n):    
        for j in range(i+1,n):
           
            obj1 = results[i]
            ymin, xmin, ymax, xmax = obj1['bounding_box']
            xmin1 = int(xmin * CAMERA_WIDTH)
            xmax1 = int(xmax * CAMERA_WIDTH)
            ymin1 = int(ymin * CAMERA_HEIGHT)
            ymax1 = int(ymax * CAMERA_HEIGHT)
            width1 = xmax1-xmin1
            height1 = ymax1-ymin1
            area1 = width1*height1
           
            obj2 = results[j]
            ymin, xmin, ymax, xmax = obj2['bounding_box']
            xmin2 = int(xmin * CAMERA_WIDTH)
            xmax2 = int(xmax * CAMERA_WIDTH)
            ymin2 = int(ymin * CAMERA_HEIGHT)
            ymax2 = int(ymax * CAMERA_HEIGHT)
            width2 = xmax2-xmin2
            height2 = ymax2-ymin2
            area2 = width2*height2
           
            if (area1<area2):
                tem = results[i]
                results[i] = results[j]
                results[j] = tem
        ##
    ##
    #print('after sort')
    for obj in results:
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * CAMERA_WIDTH)
        xmax = int(xmax * CAMERA_WIDTH)
        ymin = int(ymin * CAMERA_HEIGHT)
        ymax = int(ymax * CAMERA_HEIGHT)
        width = xmax-xmin
        height = ymax-ymin
        area = width*height
        #print(labels[obj['class_id']], '/', area, end=' ')
    #print('')
    return results

def calculate_vibration():
    dist = distance()
    #print('Distance measured in cm: ', dist)
    vibration = 0
    
    if (dist<200.0):
        vibration = 1
    elif (dist<1000.0):
        vibration = 0.5
    return vibration

def distance():
    # set Trigger to HIGH
    GPIO.output(GPIO_TRIGGER, True)

    # set Trigger after 0.01ms to LOW
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)

    StartTime = time.time()
    StopTime = time.time()

    # save StartTime
    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()

    # save time of arrival
    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()

    # time difference between start and arrival
    TimeElapsed = StopTime - StartTime
    # multiply with the sonic speed (34300 cm/s)
    # and divide by 2, because there and back
    distance = (TimeElapsed * 34300) / 2

    return distance

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  parser.add_argument(
      '--threshold',
      help='Score threshold for detected objects.',
      required=False,
      type=float,
      default=0.4)
  args = parser.parse_args()

  labels = load_labels(args.labels)
  interpreter = Interpreter(args.model)
  interpreter.allocate_tensors()
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  with picamera.PiCamera(
      resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=30) as camera:
    camera.start_preview()
    try:
      stream = io.BytesIO()
      annotator = Annotator(camera)
      while True:
        motor.off() 
        camera.capture(stream, format='jpeg', use_video_port=False)
        stream.seek(0)
        image = Image.open(stream).convert('RGB').resize(
            (input_width, input_height), Image.ANTIALIAS)
        start_time = time.monotonic()
        results = detect_objects(interpreter, image, args.threshold)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        annotator.clear()
        annotate_objects(annotator, results, labels)
        annotator.text([5, 0], '%.1fms' % (elapsed_ms))
        annotator.update()
       
        sorted = sorting_objects(results, labels)
        vocalize_objects(sorted, labels)
       
        motor.on()
        vibration = calculate_vibration()
        motor.value = vibration
        time.sleep(1)
        
        stream.seek(0)
        stream.truncate()

    finally:
      camera.stop_preview()

try:
    if __name__ == '__main__':
        main()
except KeyboardInterrupt:
    pass




