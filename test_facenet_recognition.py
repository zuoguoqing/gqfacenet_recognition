import cv2
from test_facenet_register import FaceRecognition
from PIL import Image, ImageDraw
import multiprocessing as mp
import time

face_recognition = FaceRecognition("config_facenet.yaml")

def recognition_photo():
    frame = Image.open('datasets/multiface.jpg')
    results = face_recognition.recognition(frame)
    print(results)

    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    for result in results:
        draw.rectangle(result["bbox"], outline=(255, 255, 255))
        if len(result["userid"]) > 0:
            userid = result["userid"].split("_")
            userid.pop(len(userid) - 1)
            draw.text((int(result["bbox"][0]), int(result["bbox"][1])), str("_".join(userid)), fill=(255, 255, 255),
                      font=face_recognition.font)
        if result.get("emotion") is not None and len(result["emotion"]) > 0:
            draw.text((int(result["bbox"][0]), int(result["bbox"][1] + 20)), str(result["emotion"]),
                      fill=(255, 255, 255),
                      font=face_recognition.font)
    frame_draw.save('output/multiface_facenet.jpg')

def recognition_video():
    camara = cv2.VideoCapture(0)
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
    camara.set(cv2.CAP_PROP_FPS, 25)
    while True:
        # 读取帧摄像头
        ret, frame = camara.read()
        if ret:
            frame = cv2.flip(frame, 1)
            results = face_recognition.recognition(Image.fromarray(frame))
            print(results)
            if results is not None:
                for result in results:
                    cv2.rectangle(frame, (int(result['bbox'][0]), int(result['bbox'][1])),
                                  (int(result['bbox'][2]), int(result['bbox'][3])), (255, 255, 255), 2)
                    if len(result["userid"]) > 0:
                        userid = result["userid"].split("_")
                        userid.pop(len(userid) - 1)
                        cv2.putText(frame, str("_".join(userid)), (int(result['bbox'][0]), int(result['bbox'][1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))
                    if result.get("emotion") is not None and len(result["emotion"]) > 0:
                        cv2.putText(frame, str(result["emotion"]),
                                    (int(result['bbox'][0]), int(result['bbox'][1] + 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255, 255, 255))
            cv2.imshow('recognition_face', frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    camara.release()
    cv2.destroyAllWindows()

def camera_put(queue, url):
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
    cap.set(cv2.CAP_PROP_FPS, 25)
    if cap.isOpened():
        print(f"视频地址:{url}")
    while True:
        ret, frame = cap.read()
        if ret:
            queue.put(frame)
            time.sleep(0.01)



def camera_get(queue, winname):
    cv2.namedWindow(winname, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = queue.get()
        frame = cv2.flip(frame, 1)
        results = face_recognition.recognition(Image.fromarray(frame))
        print(results)
        if results is not None:
            for result in results:
                cv2.rectangle(frame, (int(result['bbox'][0]), int(result['bbox'][1])),
                              (int(result['bbox'][2]), int(result['bbox'][3])), (255, 255, 255), 2)
                if len(result["userid"]) > 0:
                    userid = result["userid"].split("_")
                    userid.pop(len(userid) - 1)
                    cv2.putText(frame, str("_".join(userid)), (int(result['bbox'][0]), int(result['bbox'][1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))
                if result.get("emotion") is not None and len(result["emotion"]) > 0:
                    cv2.putText(frame, str(result["emotion"]),
                                (int(result['bbox'][0]), int(result['bbox'][1] + 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255))
        cv2.imshow(winname, frame)
        cv2.waitKey(1)

def run_single_camera():
    mp.set_start_method(method='spawn')  # init
    queue = mp.Queue(maxsize=2)
    camera_url = 0
    processes = [mp.Process(target=camera_put, args=(queue, camera_url)),
                 mp.Process(target=camera_get, args=(queue, f"{camera_url}"))]

    [process.start() for process in processes]
    [process.join() for process in processes]

def run_multi_camera():

    camera_urls = [
        "rtsp://username:password@192.168.1.100/h264/ch1/main/av_stream",
        "rtsp://username:password@192.168.1.101//Streaming/Channels/1",
        "rtsp://username:password@192.168.1.102/cam/realmonitor?channel=1&subtype=0"
    ]

    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=4) for _ in camera_urls]
    processes = []
    for queue, camera_url in zip(queues, camera_urls):
        processes.append(mp.Process(target=image_put, args=(queue, camera_url)))
        processes.append(mp.Process(target=image_get, args=(queue, camera_url)))

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()


if __name__ == '__main__':
    # recognition_photo()
    # recognition_video()
    run_single_camera()