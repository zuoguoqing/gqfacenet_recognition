from deepface import DeepFace

results = DeepFace.verify([['datasets/train_images/courteney_cox/1.jpg', 'datasets/train_images/courteney_cox/2.jpg'], ['datasets/train_images/jennifer_aniston/1.jpg', 'datasets/train_images/courteney_cox/1.jpg']], model_name='Ensemble')
print("Is verified: ", results["pair_1"]["verified"], results["pair_2"]["verified"])

obj = DeepFace.analyze(img_path="datasets/train_images/jennifer_aniston/1.jpg", actions=['age', 'gender', 'race', 'emotion'])
print(obj["age"], " years old ", obj["dominant_race"], " ", obj["dominant_emotion"], " ", obj["gender"])
