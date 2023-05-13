# 1.	Работа с изображениями:
# a)	Перевод в градации серого и в чёрно-белое изображение по порогу
# b)	Добавление надписей, кадрирование, изменение размера
# c)	Поворот изображения, размытие и сглаживание.

import cv2

img = cv2.imread("cat.jpg")
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, threshold_image = cv2.threshold(gray_image, 127, 255, 0)
cv2.imshow("Gray", gray_image)
cv2.imshow("Thresholded", threshold_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cropped = img[400:800, 600:2000]
cv2.imshow("Cropped image", cropped)
scale_percent = 30
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized", resized)
output = resized.copy()
cv2.putText(output, "It is a cat.", (5,300),cv2.FONT_HERSHEY_SIMPLEX, 2, (28, 28, 28), 3)
cv2.imshow("Image with text", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

(h, w, d) = img.shape
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
cv2.imshow("Rotated image", rotated)
res = cv2.blur(img,(3,3))
cv2.imshow("Result", res)
blurred = cv2.GaussianBlur(img, (51, 51), 0)
cv2.imshow("Blurred image", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Распознавание лиц на фото. Написать алгоритм, который будет распознавать лица на фотографии, обводить их
# прямоугольными рамками и выводить количество найденных лиц.

import cv2

img = cv2.imread("people.jpg")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_image)
faces_detected = "Лиц обнаружено: " + format(len(faces))
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
print(faces_detected)
cv2.imwrite("faces_detected.jpg", img)
image = cv2.imread("faces_detected.jpg")
cv2.imshow("Detected faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. Рисование. С помощью opencv python написать функции, которые по действиям мыши будут рисовать фигуры или
# переключаться в режим свободного рисования. Необходимо:
# i.	Создать пустое окно с чёрным фоном
# ii.	Реализовать функцию построения фигуры при двойном нажатии левой кнопки мыши
# iii.	Реализовать функцию свободного рисования при нажатии и удерживании правой кнопки (Подсказка: можно
# воспользоваться многократным повторением кругов с маленьким диаметром и привязать его к движению мыши)
# iv.	*Дополнительные функции, если потребуются
# v.	Добавить подпись
# vi.	Реализовать прерывание программы и закрытие окна при нажатии клавиши ‘q’

import cv2
import numpy as np

drawing = False

def drawing_image(event, x, y, flags, param):
    global drawing

    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 60, (255, 252, 255), -1)

    if event == cv2.EVENT_RBUTTONDOWN:
        drawing = True

    if event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    if event == cv2.EVENT_RBUTTONUP:
        drawing = False
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

img = np.zeros((550, 550, 3), np.uint8)
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', drawing_image)
cv2.putText(img, "It is a snowman.", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (28, 28, 28), 2)
while True:
   cv2.imshow("Image", img)
   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cv2.destroyAllWindows()

# 2. Работа с видео.
# a) Вариант 1 - Вырезать фрагмент видео (не в секундах, а в пикселях) и повернуть его на 90 градусов вправо.
# b) Вариант 2 - Вырезать фрагмент видео, далее представить три окна: исходное видео в полном размере, фрагмент в
# версии RGB и фрагмент в оттенках серого.
# c) Вариант 3 - Уменьшить размер видео, повернуть его на 180 градусов, вывести два окна: перевёрнутое видео и его
# версию в размытии.

# вариант 1
import cv2

cap = cv2.VideoCapture("video.MP4")
while True:
    success, frame = cap.read()
    crop_img = frame[150:600, 300:750]
    frame_res = cv2.rotate(crop_img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('Video', frame_res)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# вариант 2
import cv2

cap = cv2.VideoCapture("video.MP4")
while(cap.isOpened()):
    success, frame = cap.read()
    crop_img = frame[150:600, 300:750]
    rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Video', frame)
    cv2.imshow('RGB', rgb)
    cv2.imshow('Gray', gray)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# вариант 3
import cv2

percent = 50
cap = cv2.VideoCapture("video.MP4")
while cap.isOpened():
    ret,frame = cap.read()
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    video_resize = cv2.resize(frame, dim)
    video_rot = cv2.rotate(video_resize, cv2.ROTATE_180)
    blur = cv2.GaussianBlur(video_rot, (5, 5), 0)
    cv2.imshow('Video rotated', video_rot)
    cv2.imshow('Video blurred', blur)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 3. Обнаружение движущихся объектов на видео. Задачи:
# a)	Окрыть видео в opencv
# b)	Использовать алгоритм вычитания фона (fgMask = bgSubtractor.apply) и морфологические операции для удаления шума
# c)	Настроить границы (контуры) прямоугольников или иных фигур вокруг движущихся объектов
# d)	Отобразить результат.

import cv2
import numpy as np

video = cv2.VideoCapture('video.mp4')
kernel = None
bgSubtractor = cv2.createBackgroundSubtractorMOG2()
while True:
    ret, frame = video.read()
    if not ret:
        break
    fgmask = bgSubtractor.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frameCopy = frame.copy()
    for c in contours:
        if cv2.contourArea(c) > 200:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frameCopy, (x, y), (x + w, y + h), (0, 0, 255), 2)
    foregroundPart = cv2.bitwise_and(frame, frame, mask=fgmask)
    stacked = np.hstack((frame, frameCopy))
    cv2.imshow('Original frame and detected motion', cv2.resize(stacked, None, fx=0.6, fy=0.61))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()