import cv2
import time
from utils.Util import print_camera_message, set_laptop_camera
def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def find_contour(img):
    # 读取图片测试
    # img = cv2.imread('Record_917.png')
    # img = cv2.imread('Record_66.png')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # cv_show(thresh, 'thresh')

    # 灰度图像，findcontours
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 传入绘制图像，轮廓，轮廓索引，颜色模式，线条厚度
    # 注意需要copy,要不原图会变
    draw_img = img.copy()
    # draw_img = img
    res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
    # cv_show(res, 'res')

    return contours

def filtrate_contour(img,contours):
    # 检测直立车

    # 首先取出所有的灯的边界
    xywh_everycontour = []
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        xywh_everycontour.append([x, y, w, h])

    #     经检查侧翻的车的灯无法识别（917）
    #     img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #     cv_show(img,'img')

    # print(xywh_everycontour)

    # 对灯的宽高比 以大致范围 进行筛选，得到所有 可能是 装甲板旁侧灯的位置
    board_light = []
    for i in xywh_everycontour:
        if 4 > i[3] / i[2] > 2:
            board_light.append(i)
            # print(xywh_everycontour.index(i))
            x, y, w, h = xywh_everycontour[xywh_everycontour.index(i)]

            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # print(board_light)
    # print('********************************')
    # 按照 不同组旁侧灯 的比例相同来筛选
    match = []
    for i in range(len(board_light)):
        for j in range(i + 1, len(board_light)):
            if abs((board_light[i][2] * board_light[i][3]) - (board_light[j][2] * board_light[j][3])) / (
                    board_light[i][2] * board_light[i][3]) < 0.2:
                match.append([board_light[i], board_light[j]])
        #     print(match)
        # print('********************************')
    # print(match)
    # 在可能匹配的灯组中根据 xy坐标信息 进行筛选
    board = []
    for i in range(len(match)):
        #   以两侧灯的 y的偏差 来排除非平行
        if abs(match[i][0][1] - match[i][1][1]) / match[i][0][1] < 0.05:
            board.append(match[i])

    # print(board)

    for i in board:
        # print(i[0], i[1])
        if i[0][0] > i[1][0]:
            x, y, w, h = i[1][0] + i[1][2], i[1][1], abs(i[0][0] - i[1][0] - i[1][2]), int((i[1][3] + i[0][3]) / 2)
        else:
            x, y, w, h = i[0][0] + i[0][2], i[0][1], abs(i[1][0] - i[0][0] - i[0][2]), int((i[1][3] + i[0][3]) / 2)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv_show(img, 'img')

    return img

    # 对单张图片的操作
    # plt.subplot()
    # plt.xticks([]), plt.yticks([])
    # plt.imshow(img)
    # plt.savefig('board')

    # ( 4 , 5 ) [1075, 596, 6, 21], [1025, 596, 7, 21]  ( 9 , 10 )  [633, 543, 9, 19], [709, 540, 8, 20]

    # x,y,w,h = xywh_everycontour[0]
    # img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    # cv_show(img, 'img')


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# set_pcb_camera(cap)
# set_pcbblack_camera(cap)
# set_blackbox_camera(cap)
set_laptop_camera(cap)

ret=cap.set(6, cv2.VideoWriter.fourcc('M','J','P','G'))
ret=cap.set(5, 1000)
# time.sleep(0.5)
# （176×144）、CIF（352×288）、D1（704×576）、720P（1280×720）、1080P（1920*1080）
ret=cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
# time.sleep(0.5)
# ret=cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# time.sleep(0.5)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
ret=cap.set(cv2.CAP_PROP_EXPOSURE, -4)
ret=cap.set(cv2.CAP_PROP_BRIGHTNESS, 20)
# ret=cap.set(cv2.CAP_PROP_BRIGHTNESS, 1000)
ret=cap.set(cv2.CAP_PROP_GAIN, 50)

print_camera_message(cap)

while 1:
    time1 = time.time()
    ret, img = cap.read()
    # contours=find_contour(img)
    # img=filtrate_contour(img,contours)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xff == ord(' '):
        cap.release()
        cv2.destroyAllWindows()
        break
    time2=time.time()
    fps = 1 / (time2 - time1)
    print('帧率：' + str(fps))
    cv2.putText(img, "FPS {0}".format(fps), (10, 30), 1, 1.5, (255, 255, 255), 2)



