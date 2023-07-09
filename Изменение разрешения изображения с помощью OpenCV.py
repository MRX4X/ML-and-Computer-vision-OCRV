import cv2
import os


#Цистерны
puth_new = '/home/user/Изображения/Цистерны с измен.рес/'
puth = '/home/user/Изображения/Цистерны/'
currentDirectory = os.listdir(puth)

count = 0
for currentFile in currentDirectory:  
    src = cv2.imread(puth+currentFile, cv2.IMREAD_UNCHANGED)

    new_width = 350
    new_height = 350

    dsize = (new_width, new_height)

    output = cv2.resize(src, dsize, interpolation = cv2.INTER_AREA)

    count+=1

    cv2.imwrite(puth_new+str('Д')+str(count)+'.jpg',output) 


#Полувагоны
puth_new = '/home/user/Изображения/Полувагоны изменение рес/'
puth = '/home/user/Изображения/Полувагоны_законченный вариант/'
currentDirectory = os.listdir(puth)

count = 0
for currentFile in currentDirectory:  
    src = cv2.imread(puth+currentFile, cv2.IMREAD_UNCHANGED)

    new_width = 350
    new_height = 350

    dsize = (new_width, new_height)

    output = cv2.resize(src, dsize, interpolation = cv2.INTER_AREA)

    count+=1

    cv2.imwrite(puth_new+str('Д')+str(count)+'.jpg',output) 


#Хопперы
puth_new = '/home/user/Изображения/Хоппер с изменен рес/'
puth = '/home/user/Изображения/Хопперы/'
currentDirectory = os.listdir(puth)

count = 0
for currentFile in currentDirectory:  
    src = cv2.imread(puth+currentFile, cv2.IMREAD_UNCHANGED)

    new_width = 350
    new_height = 350

    dsize = (new_width, new_height)

    output = cv2.resize(src, dsize, interpolation = cv2.INTER_AREA)

    count+=1

    cv2.imwrite(puth_new+str('Д')+str(count)+'.jpg',output) 