from django.shortcuts import render
import tensorflow as tf
from django.http import HttpResponse
import os
import cv2
import numpy as np
from PIL import Image
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.preprocessing import image
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import InMemoryUploadedFile
from io import StringIO, BytesIO

from django.core.files import File
from matplotlib import cm
import matplotlib.pyplot as plt

# Create your views here.

img_height = 96
img_width = 128

def index(request):
    context={'a':1}
    return render(request,'predict.html',context)


def predictImage(request):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = os.path.join(os.path.dirname(BASE_DIR), 'drimg_segment', 'model', 'drone_semantic_model.hdf5')
    new_model = tf.keras.models.load_model(model)


    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    picture_name = str(fileObj.name)
    picture_name = picture_name.partition('.')
    picture_name = picture_name[0]
    print(picture_name)
    testimage='./static'+filePathName
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    #x = np.asarray(x).astype('float64')
    #x = np.asarray(x)
    img = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 96))
    img = img / 255
    img = np.array(img, dtype='float64')
    pred = new_model.predict(np.expand_dims(img, 0))
    pred_mask = np.argmax(pred, axis=-1)
    pred_mask = pred_mask[0]
    pred_mask = np.clip(pred_mask, 0, 22)
    pred_mask = (pred_mask - float(np.min(pred_mask))) / float((np.max(pred_mask)) - float(np.min(pred_mask)))
    pred_mask = cm.viridis(pred_mask)*255
    



    # a colormap and a normalization instance
    """cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=pred_mask.min(), vmax=pred_mask.max())
    # map the normalized data to colors
    # image is now RGBA (512x512x4) 
    imagined = cmap(norm(pred_mask))"""

    pred_mask = np.asarray(pred_mask, dtype='uint8')
    pred_mask = Image.fromarray(pred_mask, mode="RGBA")
    pred_mask_obj = pred_mask
    # save the image
    

    img_io = BytesIO()
    pred_mask.save(img_io, format='PNG', quality=100)
    pred_mask = ContentFile(img_io.getvalue(), 'pred.jpg')

    
    pred_image_filepath=fs.save(picture_name + '_pred.jpg',pred_mask)
    pred_image_filepath=fs.url(pred_image_filepath)



    context={'filePathName':filePathName, 'predictedLabel':pred_image_filepath}
    #context={'filePathName':filePathName,}
    #context={'filePathName':filePathName, 'predictedLabel':pred_mask}
    return render(request,'predict.html',context) 

def viewDataBase(request):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    media_folder = os.path.join(os.path.dirname(BASE_DIR), 'drimg_segment', 'static', 'media')
    listOfImages=os.listdir(media_folder)
    listOfImagesPath=['./media/'+i for i in listOfImages if '_pred' not in i]
    listOfSegmentationsPath=['./media/'+i for i in listOfImages if '_pred' in i]
    print(listOfImagesPath)
    print(listOfSegmentationsPath)
    context={'listOfImagesPath':listOfImagesPath, 'listOfSegmentationsPath':listOfSegmentationsPath,}
    #return HttpResponse('Hello')
    return render(request,'viewDB.html',context) 




def predict(request):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = os.path.join(os.path.dirname(BASE_DIR), 'drimg_segment', 'model', 'drone_semantic_model.hdf5')
    new_model = tf.keras.models.load_model(model)

    """if request.method == "POST":
        form = PredictionForm(request.POST, request.FILES)
        if form.is_valid():
            print (request.POST.dict())
            print(request.FILES.dict())
            fileObj=request.FILES['image']
            fs=FileSystemStorage()
            filePathName=fs.save(fileObj.name,fileObj)
            testimage='.'+filePathName
            #image = form.cleaned_data.get("image")
            print(filePathName)
            image = cv2.imread(testimage)
            print(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (128, 96))
            pred = new_model.predict(np.expand_dims(image, 0))
            pred_mask = np.argmax(pred, axis=-1)
            pred_mask = pred_mask[0]
            obj = Predictions.objects.create(
                image = image,
                pred_mask = pred_mask,
                 )
            obj.save()
            print(obj)
    else:
        form = PredictionForm()

    context = {'form': form,}"""
    return render(request, 'predict.html', context)


def predict_chances(request):

    if request.POST.get('action') == 'post':

        # Receive data from client
        image = request.POST.get('image')
        

        # Unpickle model
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model = os.path.join(os.path.dirname(BASE_DIR), 'drimg_segment', 'model', 'drone_semantic_model.hdf5')
        new_model = tf.keras.models.load_model(model)
        #HEROKU UPLOAD
        #pickle_model = os.path.join(os.path.dirname(BASE_DIR), 'app', 'model.pickle')
        #files = os.listdir(BASE_DIR)
        #print("Files in %r: %s" % (BASE_DIR, files))
        

        return JsonResponse({'result': 'Hello',
                             },
                            safe=False)

        """return JsonResponse({'title': title,},
                            safe=False)"""

        #return HttpResponse('Hello')

def view_results(request):
    return HttpResponse('Hello')
    # Submit prediction and show all
    """data = {"dataset": Predictions.objects.all()}
    return render(request, "results.html", data)"""