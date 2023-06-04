from django.shortcuts import render
from django.http.response import JsonResponse
from rest_framework.parsers import JSONParser
from rest_framework import status

from itti.models import Saliency
from itti.serializers import SaliencySerializer
from rest_framework.decorators import api_view

import cv2
from itti.pySaliencyMap import pySaliencyMap
import urllib.request
import numpy as np
from PIL import Image, ImageChops 
import matplotlib.pyplot as plt


# Create your views here.


@api_view(['POST'])
def image_list_arriba(request):

    # Recupere todos los tutoriales / busque por titulo de la base de datos de MongoDB:
    # if request.method == 'GET':
    #     saliencys = Saliency.objects.all()
    #     title = request.GET.get('title', None)
    #     if title is not None:
    #         saliencys = saliencys.filter(title__icontains=title)
    #     saliencys_serializer = SaliencySerializer(saliencys, many=True)
    #     return JsonResponse(saliencys_serializer.data, safe=False)

    # Crear y guardar un nuevo tutorial:
    if request.method == 'POST':
        vectorimage_data = JSONParser().parse(request)
        urlImage = vectorimage_data['urlImage']

        # READ
        url_response = urllib.request.urlopen(urlImage)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)

        def convert_from_cv2_to_image(img: np.ndarray) -> Image:
            # return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return Image.fromarray(img)
        
        def trim(im): 
            bg = Image.new(im.mode, im.size, im.getpixel((0,0))) 
            diff = ImageChops.difference(im, bg) 
            diff = ImageChops.add(diff, diff, 2.0, -100)
            bbox = diff.getbbox() 
            if bbox: 
                return im.crop(bbox)

        pilImage = convert_from_cv2_to_image(img)
        # pilImage.show()
        img11 = trim(pilImage)
        # img11.show()

        pil_image11 = img11.convert('RGB') 
        img = np.array(pil_image11) 
        # cv2.imshow("input",  img)

        # INITIALIZE
        imgsize = img.shape
        img_width = imgsize[1]  # img_width = 864
        img_height = imgsize[0]  # img_height = 1080

        sm = pySaliencyMap(img_width, img_height)  # def __init__

        # computation
        saliency_map = sm.SMGetSM_Arriba(img)


        vectorimage_data['vectorImage'] = saliency_map

        saliency_serializer = SaliencySerializer(data=vectorimage_data)
        print('saliency_serializer', saliency_serializer)

        if saliency_serializer.is_valid():
            saliency_serializer.save()
            return JsonResponse(saliency_serializer.data, status=status.HTTP_201_CREATED)
        return JsonResponse(saliency_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def image_list_abajo(request):

    # Crear y guardar un nuevo tutorial:
    if request.method == 'POST':
        vectorimage_data = JSONParser().parse(request)
        urlImage = vectorimage_data['urlImage']

        # READ
        url_response = urllib.request.urlopen(urlImage)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)

        def convert_from_cv2_to_image(img: np.ndarray) -> Image:
            # return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return Image.fromarray(img)
        
        def trim(im): 
            bg = Image.new(im.mode, im.size, im.getpixel((0,0))) 
            diff = ImageChops.difference(im, bg) 
            diff = ImageChops.add(diff, diff, 2.0, -100)
            bbox = diff.getbbox() 
            if bbox: 
                return im.crop(bbox)

        pilImage = convert_from_cv2_to_image(img)
        # pilImage.show()
        img11 = trim(pilImage)
        # img11.show()

        pil_image11 = img11.convert('RGB') 
        img = np.array(pil_image11) 
        # cv2.imshow("input",  img)


        # INITIALIZE
        imgsize = img.shape
        img_width = imgsize[1]  # img_width = 864
        img_height = imgsize[0]  # img_height = 1080

        sm = pySaliencyMap(img_width, img_height)  # def __init__

        # computation
        saliency_map = sm.SMGetSM_Abajo(img)
        print('saliency_map', saliency_map)

        vectorimage_data['vectorImage'] = saliency_map

        saliency_serializer = SaliencySerializer(data=vectorimage_data)


        if saliency_serializer.is_valid():
            saliency_serializer.save()
            return JsonResponse(saliency_serializer.data, status=status.HTTP_201_CREATED)
        return JsonResponse(saliency_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def image_list_full(request):

    # Crear y guardar un nuevo tutorial:
    if request.method == 'POST':
        vectorimage_data = JSONParser().parse(request)
        urlImage = vectorimage_data['urlImage']

        # READ
        url_response = urllib.request.urlopen(urlImage)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)

        def convert_from_cv2_to_image(img: np.ndarray) -> Image:
            # return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return Image.fromarray(img)
        
        def trim(im): 
            bg = Image.new(im.mode, im.size, im.getpixel((0,0))) 
            diff = ImageChops.difference(im, bg) 
            diff = ImageChops.add(diff, diff, 2.0, -100)
            bbox = diff.getbbox() 
            if bbox: 
                return im.crop(bbox)

        pilImage = convert_from_cv2_to_image(img)
        # pilImage.show()
        img11 = trim(pilImage)
        # img11.show()

        pil_image11 = img11.convert('RGB') 
        img = np.array(pil_image11) 
        # cv2.imshow("input",  img)


        # INITIALIZE
        imgsize = img.shape
        img_width = imgsize[1]  # img_width = 864
        img_height = imgsize[0]  # img_height = 1080

        sm = pySaliencyMap(img_width, img_height)  # def __init__

        # computation
        saliency_map = sm.SMGetSM_Full(img)

        vectorimage_data['vectorImage'] = saliency_map

        saliency_serializer = SaliencySerializer(data=vectorimage_data)

        if saliency_serializer.is_valid():
            saliency_serializer.save()
            return JsonResponse(saliency_serializer.data, status=status.HTTP_201_CREATED)
        return JsonResponse(saliency_serializer.errors, status=status.HTTP_400_BAD_REQUEST)