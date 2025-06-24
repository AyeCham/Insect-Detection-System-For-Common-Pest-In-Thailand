import cv2
import math
import datetime
from ultralytics import YOLO
from gpiozero import LED
from picamera2 import Picamera2
import os
import requests
import csv

# === Setup ===
led = LED(15)
model = YOLO("/home/admin/yolo/30classes/best30.pt")
classNames = ["Aphids", "Beet Weevil", "Blister Beetle", "Brown Plant Hopper", "Carriola", "Durian Borer", "Flea Beetle", "Fruit Flies", "Glenea", "Legume Blister Beetle", "Longlegged Spider Mite", "Mango Leafhopper", "Mango-Leaf-Twister", "Mealybug", "Paddy Stem Maggot", "Psyllids", "Rice Gall Midge", "Rice Leaf Caterpillar", "Rice Leaf Roller", "Rice Leafhopper", "Rice Shell Pest", "Rice Water Weevil", "Scirtothrips dorsalis Hood", "Small Brown Plant Hopper", "Thrips"]

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1384791732142608404/FMCuVrVKhHTSbrNw-JZ4Ji6KO1ZU8Kh2x29Y_GHnJDTTGuYANR0LIjG-PGbPZpwJzCSN"

log_file_path = "/home/admin/yolo/detected/detections_log.csv"

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)}, buffer_count=3)
picam2.configure(config)
picam2.start()



# Create log file with header if not exists
if not os.path.exists(log_file_path):
    with open(log_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Species", "Confidence", "ImagePath", "x1", "y1", "x2", "y2"])

        
# === Upload to Google Drive ===
# def upload_to_drive(file_name, file_path, folder_id):
#     try:
#         SCOPES = ['https://www.googleapis.com/auth/drive.file']
#         credentials = service_account.Credentials.from_service_account_file(
#             SERVICE_ACCOUNT_JSON, scopes=SCOPES)

#         service = build('drive', 'v3', credentials=credentials)

#         file_metadata = {'name': file_name, 'parents': [folder_id]}
#         media = MediaFileUpload(file_path, resumable=True)
#         file = service.files().create(
#             body=file_metadata,
#             media_body=media,
#             fields='id'
#         ).execute()

#         print(f' Uploaded to Google Drive. File ID: {file.get("id")}')
#     except Exception as e:
#         print(f" Error uploading to Google Drive: {e}")



# === Discord Alert Function ===
def send_discord_alert(message, image_path=None):
    try:
        data = {"content": message}
        if image_path:
            with open(image_path, 'rb') as f:
                files = {"file": (os.path.basename(image_path), f.read())}
                response = requests.post(DISCORD_WEBHOOK_URL, data=data, files=files)
        else:
            response = requests.post(DISCORD_WEBHOOK_URL, data=data)

        if response.status_code in (200, 204):
            print(" Discord alert sent.")
        else:
            print(f" Discord alert failed: {response.status_code}, {response.text}")

    except Exception as e:
        print(f" Discord alert error: {e}")



# === Continuous Inference ===
os.makedirs("/home/admin/yolo/detected", exist_ok=True)
try:
    print(" Starting real-time pest monitoring...")
    led.on()
    while True:
        img = picam2.capture_array()

        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        results = model(img, stream=True)
        detected_count = 0
        detected_pests = set()
        now = datetime.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        filename = f"detection_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        image_path = f"/home/admin/yolo/detected/{filename}"

        for r in results:
            for box in r.boxes:
                confidence = float(box.conf[0])
                if confidence > 0.7:
                    detected_count += 1
                    cls = int(box.cls[0])
                    label = classNames[cls] if cls < len(classNames) else "Unknown"
                    detected_pests.add(label)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label_text = f"{label} {confidence * 100:.1f}%"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Log to CSV
                    with open(log_file_path, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            current_time, label, f"{confidence:.2f}", image_path,
                            x1, y1, x2, y2  # ← NEW fields added
                        ])
        cv2.putText(img, f"Time: {current_time}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"Detections: {detected_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if detected_count > 0:
            print(f" Detected {detected_count} pest(s) at {current_time}")
            try:
                retval, buffer = cv2.imencode('.jpg', img)
                with open(image_path, 'wb') as f:
                    f.write(buffer.tobytes())
                print(f" Image saved locally: {image_path}")

                pest_list = ', '.join(sorted(detected_pests))
                alert_message = f"🚨 Pest Alert!\nDetected {detected_count} pest(s) at {current_time}\nSpecies: {pest_list}"
                send_discord_alert(alert_message, image_path=image_path)

            except Exception as e:
                print(f" Error saving image: {e}")
        else:
            print(f" No pest detected at {current_time}")

except KeyboardInterrupt:
    print(" Stopped by user.")
    led.off()
finally:
    picam2.stop()
    led.off()
    picam2.close()
