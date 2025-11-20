# see for reference: https://docs.arduino.cc/tutorials/nicla-vision/blob-detection/
#
# desktop-bounding-boy.py:
# uses blob detection to create bounding boxes around (hopefully) mosquitos


import sensor
import time
from machine import LED

blue_led = LED("LED_BLUE")
blue_led.on()

clock = time.clock()

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)

mosquitoThresholds = [
     (26, 0, -128, 127, -128, 127)
     ]

green_led = LED("LED_GREEN")

blue_led.off()
while True:
    clock.tick()

    img = sensor.snapshot()

    blobs = img.find_blobs(
        mosquitoThresholds,
        area_threshold=2000,
        merge=True
    )

    if len(blobs) > 0:
        green_led.on()
#        print(1)
    else:
        green_led.off()
#        print(0)


    for blob in blobs:
        img.draw_rectangle(blob.rect(), color=(0,255,0))
