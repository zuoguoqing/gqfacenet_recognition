import glob
import os
import uuid

import cv2
import insightface
import numpy as np
import torch
import yaml
from sklearn import preprocessing
from paz.pipelines import DetectMiniXceptionFER
from PIL import ImageFont, Image
import detection_profile
import detection_eyeblink_mouthopen
import detection_emotion
import detection_liveness
profile_detector = detection_profile.detect_face_orientation()
emotion_detector = detection_emotion.predict_emotions()
blink_detector = detection_eyeblink_mouthopen.eyeblink_mouthopen_detector()

# Deploy Configuration File Parser
class DeployConfig:
    def __init__(self, conf_file):
        if not os.path.exists(conf_file):
            raise Exception('Config file path [%s] invalid!' % conf_file)

        with open(conf_file) as fp:
            configs = yaml.load(fp, Loader=yaml.FullLoader)
            deploy_conf = configs["FACE"]
            # 正数为GPU的ID，负数为使用CPU
            self.gpu_id = deploy_conf["GPU_ID"]
            self.face_db = deploy_conf["FACE_DB"]
            self.threshold = deploy_conf["THRESHOLD"]
            self.nms = deploy_conf["NMS"]

class FaceRecognition:
    @staticmethod
    def bboverlab(boxi, boxj):
        x1 = boxi[0]
        y1 = boxi[1]
        w1 = boxi[2] - boxi[0]
        h1 = boxi[3] - boxi[1]
        x2 = boxj[0]
        y2 = boxj[1]
        w2 = boxj[2] - boxj[0]
        h2 = boxj[3] - boxj[1]
        if x1 > x2 + w2:
            return False
        elif y1 > y2 + h2:
            return False
        elif x1 + w1 < x2:
            return False
        elif y1 + h1 < y2:
            return False
        return True

    def __init__(self, conf_file):
        self.config = DeployConfig(conf_file)
        # 加载人脸识别模型
        self.model = insightface.app.FaceAnalysis(det_name='retinaface_r50_v1', rec_name='arcface_r100_v1', ga_name=None)
        self.font = ImageFont.truetype("fonts/simfang.ttf", 20, encoding="utf-8")
        self.detect_emotion = DetectMiniXceptionFER([0.1, 0.1])
        self.model.prepare(ctx_id=self.config.gpu_id, nms=self.config.nms)
        # 人脸库的人脸特征
        if os.path.exists("model/db_insightface.pt"):
            self.faces_embeddings = torch.load("model/db_insightface.pt")
        else:
            self.faces_embeddings = list()


    # 加载人脸库中的人脸
    def load_faces(self, face_db_path):
        if not os.path.exists(face_db_path):
            os.makedirs(face_db_path)
        exist_userids = list()
        files = glob.glob(face_db_path+"/*.jpg")
        for faces_embedding in self.faces_embeddings:
            exist_userids.append(faces_embedding["userid"])
        for file in files:
            userid = file.split("/")[1].split(".")[0]
            if userid not in exist_userids:
                input_image = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)
                face = self.model.get(input_image)[0]
                embedding = np.array(face.embedding).reshape((1, -1))
                embedding = preprocessing.normalize(embedding)
                self.faces_embeddings.append({
                    "userid": userid,
                    "feature": embedding
                })


    def recognition(self, image):
        faces = self.model.get(image)
        emotions = self.detect_emotion(image)
        results = list()
        for face in faces:
            bbox = np.array(face.bbox).astype(np.int32).tolist()
            result = detection_liveness.detect_liveness(Image.fromarray(image), bbox)
            # 获取人脸属性
            result["landmark"] = np.array(face.landmark).astype(np.int32).tolist()
            result["age"] = face.age
            gender = '男'
            if face.gender == 0:
                gender = '女'
            result["gender"] = gender
            # 开始人脸识别
            embedding = np.array(face.embedding).reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            result["userid"] = "unknown"
            for faces_embedding in self.faces_embeddings:
                if self.feature_compare(embedding, faces_embedding["feature"], self.config.threshold):
                    result["userid"] = faces_embedding["userid"]
            results.append(result)
        return results

    def register(self, image, username):
        # 加载人脸库中的人脸
        self.load_faces(self.config.face_db)
        faces = self.model.get(image)
        if len(faces) != 1:
            return None
        # 判断人脸是否存在
        embedding = np.array(faces[0].embedding).reshape((1, -1))
        embedding = preprocessing.normalize(embedding)
        for faces_embedding in self.faces_embeddings:
            if self.feature_compare(embedding, faces_embedding["feature"], self.config.threshold):
                return faces_embedding['userid']

        uuidstring = ''.join(str(uuid.uuid4()).split('-'))
        userid = username+'_'+uuidstring
        # 符合注册条件保存图片，同时把特征添加到人脸特征库中
        cv2.imencode('.jpg', image)[1].tofile(os.path.join(self.config.face_db, '%s.jpg' % userid))
        self.faces_embeddings.append({
            "userid": userid,
            "feature": embedding
        })
        return userid

    @staticmethod
    def feature_compare(feature1, feature2, threshold):
        diff = np.subtract(feature1, feature2)
        dist = np.sum(np.square(diff), 1)
        if dist < threshold:
            return True
        else:
            return False

if __name__ == "__main__":
    filepaths = glob.glob('datasets/train_images/*/1.jpg')
    face_recognition = FaceRecognition("config_insightface.yaml")

    for filepath in filepaths:
        img = cv2.imread(filepath)
        userid = face_recognition.register(img, filepath.split('/')[2])
        print(userid)
    torch.save(face_recognition.faces_embeddings, 'model/db_insightface.pt')
