# import cv2
# import cv2 as cv
# import serial
# import numpy as np
# from utils import Util
# import copy
# from PIL import Image, ImageDraw, ImageFont
#
# def serial_open(serialPort,baudRate):
#     # 打开串口
#     ser = serial.Serial(serialPort, baudRate, parity=serial.PARITY_ODD, stopbits=serial.STOPBITS_TWO,
#                         bytesize=serial.EIGHTBITS)
#     print(serialPort,baudRate)
#     print("参数设置：串口={{}} ，波特率={{}}".format(serialPort, baudRate))
#     return ser
#
# ser = serial_open("COM6",115200)
#
# hp_time = []
# font = ImageFont.truetype('simsun.ttc', 35)
# screen = cv.imread('../images/screen.png')
# init_screen = copy.deepcopy(screen)
#
# def init_hp():
#     global hp_red_hero1, hp_red_engineer, hp_red_infantry3, hp_red_infantry4, hp_red_infantry5, hp_red_guard, hp_red_base, hp_blue_hero1, hp_blue_engineer, hp_blue_infantry3, hp_blue_infantry4, hp_blue_infantry5, hp_blue_guard, hp_blue_base
#     hp_blue_hero1 = b'\x00'
#     hp_blue_engineer = b'\x00'
#     hp_blue_infantry3 = b'\x00'
#     hp_blue_infantry4 = b'\x00'
#     hp_blue_infantry5 = b'\x00'
#     hp_blue_guard = b'\x00'
#     hp_blue_base = b'\x00'
#     hp_red_hero1 = b'\x00'
#     hp_red_engineer = b'\x00'
#     hp_red_infantry3 = b'\x00'
#     hp_red_infantry4 = b'\x00'
#     hp_red_infantry5 = b'\x00'
#     hp_red_guard = b'\x00'
#     hp_red_base = b'\x00'
#
# def init_time():
#     global timehead, timetail
#     timehead = b'\x00'
#     timetail = b'\x00'
#
# def draw_rec(img):
#     cv2.rectangle(screen,(580,11),(800,28),color=(0,255,0),thickness=1)
#     cv2.rectangle(screen, (580, 43), (800, 60), color=(0, 255, 0), thickness=1)
#     cv2.rectangle(screen, (580, 80), (800, 97), color=(0, 255, 0), thickness=1)
#     cv2.rectangle(screen, (580, 115), (800, 132), color=(0, 255, 0), thickness=1)
#     cv2.rectangle(screen, (580, 147), (800, 164), color=(0, 255, 0), thickness=1)
#     cv2.rectangle(screen, (580, 183), (800, 200), color=(0, 255, 0), thickness=1)
#     cv2.rectangle(screen, (580, 218), (800, 235), color=(0, 255, 0), thickness=1)
#
#     cv2.rectangle(screen, (960, 11), (1180, 28), color=(0, 255, 0), thickness=1)
#     cv2.rectangle(screen, (960, 43), (1180, 60), color=(0, 255, 0), thickness=1)
#     cv2.rectangle(screen, (960, 80), (1180, 97), color=(0, 255, 0), thickness=1)
#     cv2.rectangle(screen, (960, 115), (1180, 132), color=(0, 255, 0), thickness=1)
#     cv2.rectangle(screen, (960, 147), (1180, 164), color=(0, 255, 0), thickness=1)
#     cv2.rectangle(screen, (960, 183), (1180, 200), color=(0, 255, 0), thickness=1)
#     cv2.rectangle(screen, (960, 218), (1180, 235), color=(0, 255, 0), thickness=1)
#
# def draw_hp(img,hp_bag):
#     (r1, r2, r3, r4, r5, r7, rb, b1, b2, b3, b4, b5, b7, bb) = hp_bag
#     xr1 = (int(r1[0]) / 600) * 220 + 580
#     xr2 = (int(r2[0]) / 600) * 220 + 580
#     xr3 = (int(r3[0]) / 600) * 220 + 580
#     xr4 = (int(r4[0]) / 600) * 220 + 580
#     xr5 = (int(r5[0]) / 600) * 220 + 580
#     xr7 = (int(r7[0]) / 600) * 220 + 580
#     xrb = (int(rb[0]) / 5000) * 220 + 580
#     xb1 = (int(b1[0]) / 600) * 220 + 960
#     xb2 = (int(b2[0]) / 600) * 220 + 960
#     xb3 = (int(b3[0]) / 600) * 220 + 960
#     xb4 = (int(b4[0]) / 600) * 220 + 960
#     xb5 = (int(b5[0]) / 600) * 220 + 960
#     xb7 = (int(b7[0]) / 600) * 220 + 960
#     xbb = (int(bb[0]) / 5000) * 220 + 960
#
#     cv.fillPoly(img, np.array([[[580, 11], [580, 28], [xr1,28], [xr1,11]]],dtype=int), (0, 0, 255))
#     cv.fillPoly(img, np.array([[[580, 43], [580, 60], [xr2,60], [xr2,43]]],dtype=int), (0, 0, 255))
#     cv.fillPoly(img, np.array([[[580, 80], [580, 97], [xr3,97], [xr3,80]]],dtype=int), (0, 0, 255))
#     cv.fillPoly(img, np.array([[[580, 115], [580, 132], [xr4,132], [xr4,115]]],dtype=int), (0, 0, 255))
#     cv.fillPoly(img, np.array([[[580, 147], [580, 164], [xr5,164], [xr5,147]]],dtype=int), (0, 0, 255))
#     cv.fillPoly(img, np.array([[[580, 183], [580, 200], [xr7,200], [xr7,183]]],dtype=int), (0, 0, 255))
#     cv.fillPoly(img, np.array([[[580, 218], [580, 235], [xrb,235], [xrb,218]]],dtype=int), (0, 0, 255))
#
#     cv.fillPoly(img, np.array([[[960, 11], [960, 28], [xb1, 28], [xb1, 11]]],dtype=int), (0, 0, 255))
#     cv.fillPoly(img, np.array([[[960, 43], [960, 60], [xb2, 60], [xb2, 43]]],dtype=int), (0, 0, 255))
#     cv.fillPoly(img, np.array([[[960, 80], [960, 97], [xb3, 97], [xb3, 80]]],dtype=int), (0, 0, 255))
#     cv.fillPoly(img, np.array([[[960, 115], [960, 132], [xb4, 132], [xb4, 115]]],dtype=int), (0, 0, 255))
#     cv.fillPoly(img, np.array([[[960, 147], [960, 164], [xb5, 164], [xb5, 147]]],dtype=int), (0, 0, 255))
#     cv.fillPoly(img, np.array([[[960, 183], [960, 200], [xb7, 200], [xb7, 183]]],dtype=int), (0, 0, 255))
#     cv.fillPoly(img, np.array([[[960, 218], [960, 235], [xbb, 235], [xbb, 218]]],dtype=int), (0, 0, 255))
#
#     cv.putText(img, "%s" % (r1[0]), (590, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     cv.putText(img, "%s" % (r2[0]), (590, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     cv.putText(img, "%s" % (r3[0]), (590, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     cv.putText(img, "%s" % (r4[0]), (590, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     cv.putText(img, "%s" % (r5[0]), (590, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     cv.putText(img, "%s" % (r7[0]), (590, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     cv.putText(img, "%s" % (rb[0]), (590, 233), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#     cv.putText(img, "%s" % (b1[0]), (970, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     cv.putText(img, "%s" % (b2[0]), (970, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     cv.putText(img, "%s" % (b3[0]), (970, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     cv.putText(img, "%s" % (b4[0]), (970, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     cv.putText(img, "%s" % (b5[0]), (970, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     cv.putText(img, "%s" % (b7[0]), (970, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     cv.putText(img, "%s" % (bb[0]), (970, 233), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
# def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
#     if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
#         img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     # 创建一个可以在给定图像上绘图的对象
#     draw = ImageDraw.Draw(img)
#     # 字体的格式
#     fontStyle = ImageFont.truetype(
#         "simsun.ttc", textSize, encoding="utf-8")
#     # 绘制文本
#     draw.text((left, top), text, textColor, font=fontStyle)
#     # 转换回OpenCV格式
#     return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
#
# def time_atten(img, timehead, timetail):
#     time = int( (int(timehead[0]) << 8) + int(timetail[0]) )
#     img = cv2ImgAddText(img, "%s min %s %s" % (time//60,time%60,time), 100, 50,  textColor=(255, 0, 0), textSize=50)
#     if 359 < time < 380:
#         img = cv2ImgAddText(img, "%s%s" % (time-359,'s后小能量机关可激活'), 100, 120, textColor=(0, 255, 0), textSize=20)
#     if 240 < time < 260:
#         img = cv2ImgAddText(img, "%s%s" % (time - 240,'s后小能量机关不可激活'), 100, 120, textColor=(0, 255, 0), textSize=20)
#     if 180 < time < 200:
#         img = cv2ImgAddText(img, "%s%s" % (time-180,'s后大能量机关可激活'), 100, 120, textColor=(0, 255, 0), textSize=20)
#         img = cv2ImgAddText(img, "%s%s" % (time-180,'s后第二次掉矿'), 100, 170, textColor=(0, 255, 0), textSize=20)
#     return img
#
# init_hp()
# init_time()
# while True:
#     screen = copy.deepcopy(init_screen)
#     mess = ser.read(4)
#     print(mess)
#     if len(hp_time)==0  and mess== b'\xa5':
#         hp_time.append(mess)
#         print(hp_time)
#     elif len(hp_time)<17 and len(hp_time)>=1:
#         if hp_time[0]== b'\xa5':
#             hp_time.append(mess)
#     elif len(hp_time)==17:
#         hp_blue_hero1 = hp_time[1]
#         hp_blue_engineer = hp_time[2]
#         hp_blue_infantry3 = hp_time[3]
#         hp_blue_infantry4 = hp_time[4]
#         hp_blue_infantry5 = hp_time[5]
#         hp_blue_guard = hp_time[6]
#         hp_blue_base = hp_time[7]
#         hp_red_hero1 = hp_time[8]
#         hp_red_engineer = hp_time[9]
#         hp_red_infantry3 = hp_time[10]
#         hp_red_infantry4 = hp_time[11]
#         hp_red_infantry5 = hp_time[12]
#         hp_red_guard = hp_time[13]
#         hp_red_base = hp_time[14]
#         timehead = hp_time[15]
#         timetail = hp_time[16]
#         print(hp_time)
#         hp_time = []
#
#     draw_hp(screen,(hp_red_hero1,hp_red_engineer,hp_red_infantry3,hp_red_infantry4,hp_red_infantry5,hp_red_guard,hp_red_base,hp_blue_hero1,hp_blue_engineer,hp_blue_infantry3,hp_blue_infantry4,hp_blue_infantry5,hp_blue_guard,hp_blue_base))
#     draw_rec(screen)
#     screen = time_atten(screen, timehead, timetail)
#     Util.screen_show('hp_remind', screen)
#     cv.waitKey(1)
#
#
import cv2
import cv2 as cv
import serial
import numpy as np
from utils import Util
import copy
from PIL import Image, ImageDraw, ImageFont

def serial_open(serialPort,baudRate):
    # 打开串口
    ser = serial.Serial(serialPort, baudRate, parity=serial.PARITY_ODD, stopbits=serial.STOPBITS_TWO,
                        bytesize=serial.EIGHTBITS)
    print(serialPort,baudRate)
    print("参数设置：串口={{}} ，波特率={{}}".format(serialPort, baudRate))
    return ser

ser = serial_open("COM6",115200)

hp_time = []
font = ImageFont.truetype('simsun.ttc', 35)
screen = cv.imread('../images/screen.png')
init_screen = copy.deepcopy(screen)

def init_hp():
    global hp_red_hero1, hp_red_engineer, hp_red_infantry3, hp_red_infantry4, hp_red_infantry5, hp_red_guard, hp_red_base, hp_blue_hero1, hp_blue_engineer, hp_blue_infantry3, hp_blue_infantry4, hp_blue_infantry5, hp_blue_guard, hp_blue_base
    hp_blue_hero1 = b'\x00'
    hp_blue_engineer = b'\x00'
    hp_blue_infantry3 = b'\x00'
    hp_blue_infantry4 = b'\x00'
    hp_blue_infantry5 = b'\x00'
    hp_blue_guard = b'\x00'
    hp_blue_base = b'\x00'
    hp_red_hero1 = b'\x00'
    hp_red_engineer = b'\x00'
    hp_red_infantry3 = b'\x00'
    hp_red_infantry4 = b'\x00'
    hp_red_infantry5 = b'\x00'
    hp_red_guard = b'\x00'
    hp_red_base = b'\x00'

def init_time():
    global time
    time = b'\x00'

def draw_rec(img):
    cv2.rectangle(screen,(580,11),(800,28),color=(0,255,0),thickness=1)
    cv2.rectangle(screen, (580, 43), (800, 60), color=(0, 255, 0), thickness=1)
    cv2.rectangle(screen, (580, 80), (800, 97), color=(0, 255, 0), thickness=1)
    cv2.rectangle(screen, (580, 115), (800, 132), color=(0, 255, 0), thickness=1)
    cv2.rectangle(screen, (580, 147), (800, 164), color=(0, 255, 0), thickness=1)
    cv2.rectangle(screen, (580, 183), (800, 200), color=(0, 255, 0), thickness=1)
    cv2.rectangle(screen, (580, 218), (800, 235), color=(0, 255, 0), thickness=1)

    cv2.rectangle(screen, (960, 11), (1180, 28), color=(0, 255, 0), thickness=1)
    cv2.rectangle(screen, (960, 43), (1180, 60), color=(0, 255, 0), thickness=1)
    cv2.rectangle(screen, (960, 80), (1180, 97), color=(0, 255, 0), thickness=1)
    cv2.rectangle(screen, (960, 115), (1180, 132), color=(0, 255, 0), thickness=1)
    cv2.rectangle(screen, (960, 147), (1180, 164), color=(0, 255, 0), thickness=1)
    cv2.rectangle(screen, (960, 183), (1180, 200), color=(0, 255, 0), thickness=1)
    cv2.rectangle(screen, (960, 218), (1180, 235), color=(0, 255, 0), thickness=1)

def draw_hp(img,hp_bag):
    (r1, r2, r3, r4, r5, r7, rb, b1, b2, b3, b4, b5, b7, bb) = hp_bag
    xr1 = (int(r1[0]) / 600) * 220 + 580
    xr2 = (int(r2[0]) / 600) * 220 + 580
    xr3 = (int(r3[0]) / 600) * 220 + 580
    xr4 = (int(r4[0]) / 600) * 220 + 580
    xr5 = (int(r5[0]) / 600) * 220 + 580
    xr7 = (int(r7[0]) / 600) * 220 + 580
    xrb = (int(rb[0]) / 5000) * 220 + 580
    xb1 = (int(b1[0]) / 600) * 220 + 960
    xb2 = (int(b2[0]) / 600) * 220 + 960
    xb3 = (int(b3[0]) / 600) * 220 + 960
    xb4 = (int(b4[0]) / 600) * 220 + 960
    xb5 = (int(b5[0]) / 600) * 220 + 960
    xb7 = (int(b7[0]) / 600) * 220 + 960
    xbb = (int(bb[0]) / 5000) * 220 + 960

    cv.fillPoly(img, np.array([[[580, 11], [580, 28], [xr1,28], [xr1,11]]],dtype=int), (0, 0, 255))
    cv.fillPoly(img, np.array([[[580, 43], [580, 60], [xr2,60], [xr2,43]]],dtype=int), (0, 0, 255))
    cv.fillPoly(img, np.array([[[580, 80], [580, 97], [xr3,97], [xr3,80]]],dtype=int), (0, 0, 255))
    cv.fillPoly(img, np.array([[[580, 115], [580, 132], [xr4,132], [xr4,115]]],dtype=int), (0, 0, 255))
    cv.fillPoly(img, np.array([[[580, 147], [580, 164], [xr5,164], [xr5,147]]],dtype=int), (0, 0, 255))
    cv.fillPoly(img, np.array([[[580, 183], [580, 200], [xr7,200], [xr7,183]]],dtype=int), (0, 0, 255))
    cv.fillPoly(img, np.array([[[580, 218], [580, 235], [xrb,235], [xrb,218]]],dtype=int), (0, 0, 255))

    cv.fillPoly(img, np.array([[[960, 11], [960, 28], [xb1, 28], [xb1, 11]]],dtype=int), (0, 0, 255))
    cv.fillPoly(img, np.array([[[960, 43], [960, 60], [xb2, 60], [xb2, 43]]],dtype=int), (0, 0, 255))
    cv.fillPoly(img, np.array([[[960, 80], [960, 97], [xb3, 97], [xb3, 80]]],dtype=int), (0, 0, 255))
    cv.fillPoly(img, np.array([[[960, 115], [960, 132], [xb4, 132], [xb4, 115]]],dtype=int), (0, 0, 255))
    cv.fillPoly(img, np.array([[[960, 147], [960, 164], [xb5, 164], [xb5, 147]]],dtype=int), (0, 0, 255))
    cv.fillPoly(img, np.array([[[960, 183], [960, 200], [xb7, 200], [xb7, 183]]],dtype=int), (0, 0, 255))
    cv.fillPoly(img, np.array([[[960, 218], [960, 235], [xbb, 235], [xbb, 218]]],dtype=int), (0, 0, 255))

    cv.putText(img, "%s" % (r1[0]), (590, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.putText(img, "%s" % (r2[0]), (590, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.putText(img, "%s" % (r3[0]), (590, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.putText(img, "%s" % (r4[0]), (590, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.putText(img, "%s" % (r5[0]), (590, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.putText(img, "%s" % (r7[0]), (590, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.putText(img, "%s" % (rb[0]), (590, 233), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv.putText(img, "%s" % (b1[0]), (970, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.putText(img, "%s" % (b2[0]), (970, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.putText(img, "%s" % (b3[0]), (970, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.putText(img, "%s" % (b4[0]), (970, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.putText(img, "%s" % (b5[0]), (970, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.putText(img, "%s" % (b7[0]), (970, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.putText(img, "%s" % (bb[0]), (970, 233), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def time_atten(img, time):
    time = int(time[0])
    img = cv2ImgAddText(img, "%s min %s %s" % (time//60,time%60,time), 100, 50,  textColor=(255, 0, 0), textSize=50)
    if 359 < time < 380:
        img = cv2ImgAddText(img, "%s%s" % (time-359,'s后小能量机关可激活'), 100, 120, textColor=(0, 255, 0), textSize=20)
    if 240 < time < 260:
        img = cv2ImgAddText(img, "%s%s" % (time - 240,'s后小能量机关不可激活'), 100, 120, textColor=(0, 255, 0), textSize=20)
    if 180 < time < 200:
        img = cv2ImgAddText(img, "%s%s" % (time-180,'s后大能量机关可激活'), 100, 120, textColor=(0, 255, 0), textSize=20)
        img = cv2ImgAddText(img, "%s%s" % (time-180,'s后第二次掉矿'), 100, 170, textColor=(0, 255, 0), textSize=20)
    return img

init_hp()
init_time()
while True:
    screen = copy.deepcopy(init_screen)
    mess = ser.read()
    if len(hp_time)==0  and mess== b'\xa5':
        hp_time.append(mess)
        print(hp_time)
    elif len(hp_time)<17 and len(hp_time)>=1:
        if hp_time[0]== b'\xa5':
            hp_time.append(mess)
    elif len(hp_time)==17:
        hp_blue_hero1 = hp_time[1]
        hp_blue_engineer = hp_time[2]
        hp_blue_infantry3 = hp_time[3]
        hp_blue_infantry4 = hp_time[4]
        hp_blue_infantry5 = hp_time[5]
        hp_blue_guard = hp_time[6]
        hp_blue_base = hp_time[7]
        hp_red_hero1 = hp_time[8]
        hp_red_engineer = hp_time[9]
        hp_red_infantry3 = hp_time[10]
        hp_red_infantry4 = hp_time[11]
        hp_red_infantry5 = hp_time[12]
        hp_red_guard = hp_time[13]
        hp_red_base = hp_time[14]
        time = hp_time[15]
        print(hp_time)
        hp_time = []

    draw_hp(screen,(hp_red_hero1,hp_red_engineer,hp_red_infantry3,hp_red_infantry4,hp_red_infantry5,hp_red_guard,hp_red_base,hp_blue_hero1,hp_blue_engineer,hp_blue_infantry3,hp_blue_infantry4,hp_blue_infantry5,hp_blue_guard,hp_blue_base))
    draw_rec(screen)
    screen = time_atten(screen, time)
    Util.screen_show('hp_remind', screen)
    cv.waitKey(1)




