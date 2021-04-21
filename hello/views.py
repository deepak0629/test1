from django.http import HttpResponse,JsonResponse
from django.shortcuts import render
from io import BytesIO
import numpy as np
from PIL import Image
from base64 import b64decode
from keras.models import load_model
import base64

def hello(request):
    model_name = "cifar_allpictures.h5"
    trainedModel = load_model(model_name)
    file = request.FILES['image']
    # print(base64.b64encode(file.read()))
    img = Image.open(BytesIO(b64decode(base64.b64encode(file.read()))))
    new_img = white_bg_square(img)
    resized_img = new_img.resize((32, 32), Image.ANTIALIAS)
    x_data = np.asarray(resized_img)
    x_data = x_data.astype('float32')
    x_data /= 255
    input_data = []
    input_data.append(x_data)
    predictions = trainedModel.predict_classes(np.array([x_data, ]))

    categoriesList = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    print("create label prediction")
    label = categoriesList[predictions[0]]
    print("label: " + label)
    return JsonResponse({"result": label})

def white_bg_square(img):
    "return a white-background-color image having the img in exact center"
    size = int(img.size[0]), int(img.size[1])  # (int(max(img.size)),)*2
    layer = Image.new('RGB', size, (255, 255, 255))
    imgsizeint = int(img.size[0]), int(img.size[1])
    layer.paste(img, tuple(map(lambda x: int((x[0] - x[1]) / 2), zip(size, imgsizeint))))
    return layer
