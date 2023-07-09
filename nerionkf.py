import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import cv2
import glob
from torchmetrics import Precision, Recall
import timm

recall = Recall(task="multiclass", average='macro', num_classes=3)
precision = Precision(task="multiclass", average='macro', num_classes=3)

class CustomDataset(Dataset):
    def __init__(self, images_folder, img_weight, img_height, transform=None):
        self.images_path = glob.glob(os.path.join(images_folder, f'*/*.jpg'))
        self.img_weight = img_weight
        self.img_height = img_height
        self.transform = transform
    
    
    def __len__(self):
        return len(self.images_path)
    

    def __getitem__(self, index):
        image_path = self.images_path[index]
        image_name = os.path.basename(image_path)
        
        label = 0
        label_one_hot = torch.tensor([1, 0, 0])
        
        if 'hopper' in image_name:
            label = 2 
            label_one_hot = torch.tensor([0, 0, 1])
        
        elif 'tank' in image_name:
            label = 1
            label_one_hot = torch.tensor([0, 1, 0])
            
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.img_weight, self.img_height), cv2.INTER_NEAREST)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image).permute(2, 1, 0) / 255

        return (image, label)  
    

class ConvolutionalNeuralNetwork(nn.Module):
    def init(self):
        super().init()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=48, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )

        self.drop_out = nn.Dropout()
        
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=3456, out_features=512),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU()
        )

        self.layer6 = nn.Linear(in_features=256, out_features=3)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.reshape(input=out, shape=(BATCH_SIZE, -1,))
        out = self.drop_out(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out


def train(epoch):
    model.train()
    y_gt_list = []
    y_pr_list = []
    for i, (images, labels) in enumerate(trainDataloader):
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss_list.append(loss.item())
        y_pr_list.append(outputs)
        y_gt_list.append(labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    y_pr = torch.cat(y_pr_list)
    y_gt = torch.cat(y_gt_list)
    r = recall(y_pr, y_gt)
    p = precision(y_pr, y_gt)
    print(f'recall train: {r}, precission train: {p}, F1: {(2 * r * p)/(r + p)}, epoch: {epoch}')
    return p, r, (2 * r * p)/(r + p)
        

def test(epoch):
    y_gt_list = []
    y_pr_list = []
    model.eval()
    with torch.no_grad():
        for images, labels in testDataloader:
            outputs = model(images)
            y_pr_list.append(outputs)
            y_gt_list.append(labels)

    y_pr = torch.cat(y_pr_list)
    y_gt = torch.cat(y_gt_list)
    r = recall(y_pr, y_gt)
    p = precision(y_pr, y_gt)
    print(f'recall test: {r}, precission test: {p}, F1: {(2 * r * p)/(r + p)}, epoch: {epoch}')
    torch.save(model.state_dict(), 'conv_net_model_resnet18.ckpt')
    return p, r, (2 * r * p)/(r + p)


IMAGES_FOLDER_TRAIN = '/home/user/Загрузки/Neural/Trains'
IMAGES_FOLDER_TEST = '/home/user/Загрузки/Neural/Tests'
IMG_WEIGHT = 64
IMG_HEIGHT = 64
BATCH_SIZE = 32
LEARN_RATE = 0.001
NUM_EPOCH = 20

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

trainDataset = CustomDataset(
    images_folder=IMAGES_FOLDER_TRAIN, 
    img_weight=IMG_WEIGHT, 
    img_height=IMG_HEIGHT,
    transform=ToTensor()
)

testDataset = CustomDataset(
    images_folder=IMAGES_FOLDER_TEST,
    img_weight= IMG_WEIGHT,
    img_height=IMG_HEIGHT,
    transform=ToTensor()
)

trainDataloader = DataLoader(
    dataset=trainDataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    drop_last=True
)

testDataloader = DataLoader(    
    dataset=testDataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True
)

# model = ConvolutionalNeuralNetwork().to(device=device)
model = timm.create_model('efficientnet_b7', pretrained=False, num_classes=3)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
total_step = len(trainDataloader)
loss_list = []
acc_list = []
train_f1_list = []
train_p_list = []
train_r_list = []
test_p_list = []
test_r_list = []
test_f1_list = []


for epoch in range(1, NUM_EPOCH + 1):
    train_p, train_r, train_f1 = train(epoch)
    test_p, test_r, test_f1 = test(epoch)
    train_p_list.append(train_p)
    train_r_list.append(train_r)
    train_f1_list.append(train_f1)
    test_p_list.append(test_p)
    test_r_list.append(test_r)
    test_f1_list.append(test_f1)

plt.plot(list(range(len(train_f1_list))), train_f1_list, color='r')
plt.plot(list(range(len(test_f1_list))), test_f1_list, color='b')
plt.legend(['F1_train', 'F1_test'])
plt.grid()
plt.title('f1_statistic')
plt.show()