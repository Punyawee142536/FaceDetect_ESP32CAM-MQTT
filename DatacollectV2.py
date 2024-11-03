import cv2
import urllib.request
import numpy as np
import os

# URL ของสตรีมจาก ESP32-CAM (เปลี่ยนตามที่ตั้งค่าไว้)
url = 'http://172.20.10.3/cam-lo.jpg'

# โหลดไฟล์ตรวจจับใบหน้า
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

id = input("Enter Your ID: ")
count_normal = 0
count_gray = 0
count_noise = 0
count_blur = 0
count_flip = 0
count_threshold = 0
count_contrast = 0
count_brightness = 0

# ฟังก์ชันสำหรับสร้างชื่อไฟล์ใหม่โดยไม่ต้องลบไฟล์เดิม
def get_next_filename(prefix, extension='jpg'):
    count = 1
    while os.path.exists(f'datasets2/{prefix}{count}.{extension}'):
        count += 1
    return f'datasets2/{prefix}{count}.{extension}'

# ฟังก์ชันเพิ่มนอยซ์ให้ภาพ
def add_noise(image):
    row, col = image.shape
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

# ฟังก์ชันเบลอภาพ
def blur_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred

# ฟังก์ชันกลับภาพ (flip)
def flip_image(image):
    flipped = cv2.flip(image, 1)
    return flipped

# ฟังก์ชันทำ Thresholding
def threshold_image(image):
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return thresh

# ฟังก์ชันปรับคอนทราสต์
def adjust_contrast(image, alpha=1.5):
    contrast_img = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return contrast_img

# ฟังก์ชันปรับความสว่าง
def adjust_brightness(image, beta=50):
    brightness_img = cv2.convertScaleAbs(image, alpha=1, beta=beta)
    return brightness_img

while True:
    # ดึงภาพจากสตรีมของ ESP32-CAM
    img_resp = urllib.request.urlopen(url)
    img_array = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(img_array, -1)

    # แปลงภาพเป็นขาวดำสำหรับตรวจจับใบหน้า
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        # บันทึกภาพธรรมดา
        if count_normal < 300:
            cv2.imwrite(get_next_filename(f'User.{id}.', 'jpg'), frame[y:y+h, x:x+w])
            count_normal += 1

        # บันทึกภาพ grayscale
        if count_gray < 300:
            cv2.imwrite(get_next_filename(f'User.{id}_gray.', 'jpg'), face_img)
            count_gray += 1

        # บันทึกภาพที่เพิ่มนอยซ์
        if count_noise < 200:
            noisy_img = add_noise(face_img)
            cv2.imwrite(get_next_filename(f'User.{id}_noise.', 'jpg'), noisy_img)
            count_noise += 1

        # บันทึกภาพที่เบลอ
        if count_blur < 200:
            blurred_img = blur_image(face_img)
            cv2.imwrite(get_next_filename(f'User.{id}_blur.', 'jpg'), blurred_img)
            count_blur += 1

        # บันทึกภาพที่ flip
        if count_flip < 200:
            flipped_img = flip_image(face_img)
            cv2.imwrite(get_next_filename(f'User.{id}_flip.', 'jpg'), flipped_img)
            count_flip += 1

        # บันทึกภาพที่ thresholding
        if count_threshold < 100:
            threshold_img = threshold_image(face_img)
            cv2.imwrite(get_next_filename(f'User.{id}_threshold.', 'jpg'), threshold_img)
            count_threshold += 1

        # บันทึกภาพที่ปรับคอนทราสต์
        if count_contrast < 200:
            contrast_img = adjust_contrast(face_img)
            cv2.imwrite(get_next_filename(f'User.{id}_contrast.', 'jpg'), contrast_img)
            count_contrast += 1

        # บันทึกภาพที่ปรับความสว่าง
        if count_brightness < 200:
            brightness_img = adjust_brightness(face_img)
            cv2.imwrite(get_next_filename(f'User.{id}_brightness.', 'jpg'), brightness_img)
            count_brightness += 1

        # วาดกรอบรอบใบหน้า
        cv2.rectangle(frame, (x, y), (x+w, y+h), (55, 55, 255), 2)

    # แสดงภาพที่มีการตรวจจับใบหน้า
    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)

    # หยุดการบันทึกภาพเมื่อเงื่อนไขการนับครบตามที่ตั้งไว้
    if (count_normal >= 300 and count_gray >= 300 and count_noise >= 200 and count_blur >= 200 and
        count_flip >= 200 and count_threshold >= 100 and count_contrast >= 200 and count_brightness >= 200):
        break

# ปิดหน้าต่างเมื่อเก็บข้อมูลครบ
cv2.destroyAllWindows()
print("Dataset Collection Done..................")
