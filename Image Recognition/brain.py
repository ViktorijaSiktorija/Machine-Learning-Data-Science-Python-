from imageai.Prediction import ImagePrediction
import os
execution_path = os.getcwd()

prediction = ImagePrediction()
# MobileNetV2 je pre built model, i sa ResNet od Majkrosofta isti rezultati
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join(
    execution_path, "mobilenet_v2.h5"))
prediction.loadModel()

predictions, probabilities = prediction.predictImage(
    os.path.join(execution_path, "giraffe.jpg"), result_count=5)  # result_count koliko predvidjanja oceo da nam da
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
# leopard  :  1.8417345359921455
# lynx  :  1.8286257982254028
# cheetah  :  1.8222695216536522
# jaguar  :  1.64992306381464
# impala  :  0.893397256731987
predictions, probabilities = prediction.predictImage(
    os.path.join(execution_path, "godzilla.jpg"), result_count=5)  # result_count koliko predvidjanja oceo da nam da
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
# common_iguana  :  21.154087781906128
# American_alligator  :  20.26159316301346
# pedestal  :  3.050793521106243
# triceratops  :  2.954220399260521
# African_crocodile  :  1.90314631909132
predictions, probabilities = prediction.predictImage(
    os.path.join(execution_path, "house.jpg"), result_count=5)  # result_count koliko predvidjanja oceo da nam da
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
# boathouse  :  20.541733503341675
# church  :  9.800753742456436
# palace  :  5.933235213160515
# lakeside  :  2.952353097498417
# flagpole  :  2.063644491136074
