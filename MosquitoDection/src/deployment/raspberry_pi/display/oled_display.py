# see for reference: https://github.com/mklements/OLED_Stats

import board
import digitalio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306

width = 128
height = 64

i2c = board.I2C()
oled = adafruit_ssd1306.SSD1306_I2C(width,height,i2c,addr=0x3C)

oled.fill(0)
oled.show()

image = Image.new("1",(oled.width, oled.height))

draw = ImageDraw.Draw(image)

font = ImageFont.load_default()

def getfontsize(font, text):
    left,top,right,bottom = font.getbbox(text)
    return right - left, bottom - top

text = ["TinyML", "MosquitoDetection"]

for i in range(len(text)):
    (font_width, font_height) = getfontsize(font,text[i])

    draw.text(
        (oled.width // 2 - font_width // 2, (oled.height // 2 - font_height*2 // 2) + font_height*i),
        text[i],
        font=font,
        fill=255
    )

oled.image(image)
oled.show()
