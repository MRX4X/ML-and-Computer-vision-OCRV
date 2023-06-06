import torch
import numpy as np
import warnings
import matplotlib.pyplot as plt
#Для построения моделей используется пакет torch.nn
import torch.nn as nn  
#Так как будут использоваться не линейные преобразования, нужно импортировать функции
from torch.nn import functional as F
#Также нам потребуется дата-сет, DataLoader - позволяет подгрузжать данные, также потребуется загрузить метрики, командой - pip install -q torchmetrics
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy
#Далее потребуется дата-сет, который можно взять из skalern
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm.autonotebook import tqdm

x, y = make_moons(n_samples=5000, random_state= 42, noise=.15) #n_samples - это общее количество 
#сгенерированных точек. Если аргумент представляет собой кортеж, то каждый элемент соответствует количеству
# точек в каждой луне.
#noise - Стандартное отклонение гауссовского шума, добавленного к данным.
plt.figure(figsize=(15, 6)) # создаст изображение с заданными параметрами
plt.title("moons") # Название изображения
plt.scatter(x[:, 0], x[:, 1], c=y, cmap="RdBu", alpha=.42, ec="white") # функция для построения точечной диаграммы
plt.axis("off") # установки некоторых свойств оси, которая отключенна
plt.show() # Функция для отображения графика
