import cv2 

# เปิดการใช้งานกล้อง
video = cv2.VideoCapture(0)

# โหลด Cascade Classifier สำหรับการตรวจจับใบหน้า
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# โหลดตัวจดจำใบหน้า LBPH จากไฟล์ที่บันทึกไว้
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

# ลิสต์ของชื่อที่สอดคล้องกับ serial ของใบหน้า
name_list = ["", "Bank", "Earn", "Mom", "Jom", "Aum", "jill", "Ice"]  # "" คือตัวแทนสำหรับ serial 0 (ไม่ได้กำหนดชื่อ)

while True:
    # อ่านเฟรมจากวิดีโอ
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # แปลงเป็นภาพสีเทา
    faces = facedetect.detectMultiScale(gray, 1.3, 5)  # ตรวจจับใบหน้า

    for (x, y, w, h) in faces:
        # ทำนายใบหน้าที่ตรวจจับได้
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
        
        # พิมพ์ค่าที่ทำนายได้เพื่อใช้ตรวจสอบ
        print(f"Serial: {serial}, Confidence: {conf}")

        if conf > 50:  # ลองปรับเกณฑ์ค่าความเชื่อมั่น
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,255), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50,50,255), -1)
            cv2.putText(frame, name_list[serial], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
  
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    frame=cv2.resize(frame, (640, 480))
    # แสดงผลเฟรมที่ถ่ายได้
    cv2.imshow("Frame", frame)
    
    # ตรวจสอบว่ากด 'q' เพื่อออกจากโปรแกรมหรือไม่
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

# ปิดกล้องและหน้าต่างการแสดงผล
video.release()
cv2.destroyAllWindows()
