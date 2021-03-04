from test_insightface_register import FaceRecognition
import cv2
from PIL import Image, ImageDraw

face_recognition = FaceRecognition("config_insightface.yaml")
image = cv2.imread("datasets/multiface.jpg")
results = face_recognition.recognition(image)
print(results)

frame_draw =Image.open("datasets/multiface.jpg")
draw = ImageDraw.Draw(frame_draw)
for result in results:
    draw.rectangle(result["bbox"], outline=(255, 255, 255))  # 绘制框
    if len(result["userid"]) > 0:
        userid = result["userid"].split("_")
        userid.pop(len(userid) - 1)
        draw.text((int(result["bbox"][0]), int(result["bbox"][1])), str("_".join(userid)), fill=(255, 255, 255), font=face_recognition.font)
    if result.get("emotion") is not None and len(result["emotion"]) > 0:
        draw.text((int(result["bbox"][0]), int(result["bbox"][1] + 20)), str(result["emotion"]), fill=(255, 255, 255), font=face_recognition.font)
    if result.get("age") is not None and int(result["age"]) > 0:
        draw.text((int(result["bbox"][0]), int(result["bbox"][1] + 40)), str(result["age"]), fill=(255, 255, 255), font=face_recognition.font)
    if result.get("gender") is not None and len(result["gender"]) > 0:
        draw.text((int(result["bbox"][0]), int(result["bbox"][1] + 60)), str(result["gender"]), fill=(255, 255, 255), font=face_recognition.font)
frame_draw.save('output/multiface_insightface.jpg')

camara = cv2.VideoCapture(0)
camara.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
camara.set(cv2.CAP_PROP_FPS, 25)
while True:
    # 读取帧摄像头
    ret, frame = camara.read()
    if ret:
        frame = cv2.flip(frame, 1)
        results = face_recognition.recognition(frame)
        print(results)
        if results is not None:
            for result in results:
                cv2.rectangle(frame, (int(result['bbox'][0]), int(result['bbox'][1])), (int(result['bbox'][2]), int(result['bbox'][3])), (255, 255, 255), 2)
                if len(result["userid"]) > 0:
                    userid = result["userid"].split("_")
                    userid.pop(len(userid) - 1)
                    cv2.putText(frame, str("_".join(userid)), (int(result['bbox'][0]), int(result['bbox'][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))
                if result.get("emotion") is not None and len(result["emotion"]) > 0:
                    cv2.putText(frame, str(result["emotion"]), (int(result['bbox'][0]), int(result['bbox'][1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))
        cv2.imshow('recognition_face', frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
camara.release()
cv2.destroyAllWindows()