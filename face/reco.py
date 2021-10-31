# -----检测、校验并输出结果-----
import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('PYTHON\\face_reco\\trainner\\trainner.yml')
cascade_path = 'PYTHON\\face_reco\\haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

idnum = 0
names = ['chushi', 'zyb', 'jwj', 'jyf']
justify = [0, 0]
count = 0

# 调用摄像头
print("11")
cam = cv2.VideoCapture(0)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
print("22")
while True:
    ret, img = cam.read()

    # 识别人脸
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.2,
                                          minNeighbors=5,
                                          minSize=(int(minW), int(minH)))
    if faces is not None:
        count = count + 1
        # 进行校验
        for (x, y, w, h) in faces:
            idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if idnum == 1 and confidence < 48:
                justify[0] += 1
            elif idnum == 2 and confidence < 50:
                justify[0] += 1
            else:
                justify[1] += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count > 5:
        if justify[0] == 0 and justify[1] == 0:
            continue
        elif justify[0] > justify[1]:
            print("known")
        else:
            print("unknown")
        justify[0] = 0
        justify[1] = 0
        count = 0

# 释放资源
cam.release()
cv2.destroyAllWindows()
