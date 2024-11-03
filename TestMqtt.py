import cv2
import urllib.request
import numpy as np
import paho.mqtt.client as mqtt

# การตั้งค่า MQTT
MQTT_BROKER = "broker.emqx.io"  # ใช้ EMQX broker ฟรี
MQTT_PORT = 1883
MQTT_TOPIC = "home/relay"

# สร้าง client สำหรับ MQTT
mqtt_client = mqtt.Client()

# ฟังก์ชันสำหรับเชื่อมต่อกับ MQTT broker
def connect_mqtt():
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

# ฟังก์ชันสำหรับส่งคำสั่งเปิดหรือปิดรีเลย์ผ่าน MQTT
def send_relay_command(command):
    mqtt_client.publish(MQTT_TOPIC, command)
    print(f"Sent {command} command to MQTT")

# เชื่อมต่อกับ MQTT broker
connect_mqtt()

# โหลดตัวจดจำใบหน้า LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")  # โหลดโมเดลใบหน้าที่เทรนแล้ว

# ลิสต์ของชื่อที่สอดคล้องกับ serial ของใบหน้า
name_list = ["", "Bank", "Earn", "Mom", "Jom", "Aum", "jill", "Ice"]

# โหลด Cascade Classifier สำหรับการตรวจจับใบหน้า
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# URL ของกล้อง (ถ้าคุณใช้ IP camera)
url = 'http://172.20.10.3/cam-lo.jpg'

while True:
    try:
        # ดึงภาพจาก URL
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)

        # แปลงเป็นภาพสีเทาเพื่อใช้กับ Cascade Classifier
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ตรวจจับใบหน้าในภาพ
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # ทำนายใบหน้าด้วย LBPH
            serial, conf = recognizer.predict(gray[y:y+h, x:x+w])

            # พิมพ์ค่าที่ทำนายได้เพื่อใช้ตรวจสอบ
            print(f"Serial: {serial}, Confidence: {conf}")

            if conf > 50:  # ปรับเกณฑ์ความเชื่อมั่นตามต้องการ
                name = name_list[serial]
                # แสดงกรอบและชื่อของบุคคลที่จดจำได้
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # ส่งคำสั่ง "OPEN" ไปยัง MQTT เพื่อเปิดรีเลย์
                send_relay_command("OPEN")
            else:
                # ถ้าจดจำไม่ได้ ให้แสดงว่า "Unknown"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # ส่งคำสั่ง "CLOSE" ไปยัง MQTT เพื่อปิดรีเลย์
                send_relay_command("CLOSE")

        # ปรับขนาดเฟรมและแสดงผล
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow("Frame", frame)

        # ตรวจสอบว่ากด 'q' เพื่อออกจากโปรแกรมหรือไม่
        if cv2.waitKey(1) == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        break

# ปิดการเชื่อมต่อกับ MQTT broker
mqtt_client.disconnect()

# ปิดหน้าต่างการแสดงผล
cv2.destroyAllWindows()
