import os
import torch

import gc
import cv2
import time
import datetime
import warnings
import random

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
from torchtoolbox.transform.transforms import ColorJitter, RandomApply

import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchtoolbox.transform as transforms

from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from efficientnet_pytorch import EfficientNet


warnings.simplefilter('ignore')
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(47)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MelanomaDataset(Dataset):
    def __init__(self, df, imfolder, train = True, transforms = None, meta_features = None):

        """"
        Class initialization
            Args:
                df(pd.DataFrame) = Data frame with data describe_option
                imfolder(str): location of images
                train(bool): self explanatory
                transforms: image transfromation method to be applied
                meta_features(list): list of features with meta information like sex, age
        """

        self.df = df
        self.imfolder = imfolder
        self.transforms = transforms
        self.train = train
        self.meta_features = meta_features

    def __getitem__(self, index):
        im_path = os.path.join(self.imfolder, self.df.iloc[index]['image_name'] + '.jpg')
        x = cv2.imread(im_path)
        H = 256
        x = cv2.resize(x, (H, H))
        meta = np.array(self.df.iloc[index][self.meta_features].values, dtype=np.float32)


        if(self.transforms):
            x = self.transforms(x)
        
        if(self.train):
            y = self.df.iloc[index]['target']
            return (x, meta), y
        else:
            return (x, meta)
    
    def __len__(self):
        return len(self.df)

class Net(nn.Module):
    def __init__(self, arch, n_meta_features: int):
        super(Net, self).__init__()
        self.arch = arch
        if 'ResNet' in str(arch.__class__):
            self.arch.fc = nn.Linear(in_features=512, out_features=500, bias=True)
        if 'EfficientNet' in str(arch.__class__):
            self.arch._fc = nn.Linear(in_features=1280, out_features=500, bias=True)
        self.meta = nn.Sequential(nn.Linear(n_meta_features, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(500, 250),  # FC layer output will have 250 features
                                  nn.BatchNorm1d(250),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
        self.ouput = nn.Linear(500 + 250, 1)
        
    def forward(self, inputs):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        x, meta = inputs
        cnn_features = self.arch(x)
        meta_features = self.meta(meta)
        features = torch.cat((cnn_features, meta_features), dim=1)
        output = self.ouput(features)
        return output
class AdvancedHairAugmentation:
    """
    Impose an image of a hair to the target image

    Args:
        hairs (int): maximum number of hairs to impose
        hairs_folder (str): path to the folder with hairs images
    """

    def __init__(self, hairs: int = 5, hairs_folder: str = ""):
        self.hairs = hairs
        self.hairs_folder = hairs_folder

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL Image: Image with drawn hairs.
        """
        n_hairs = random.randint(0, self.hairs)
        
        if not n_hairs:
            return img
        
        height, width, _ = img.shape  # target image width and height
        hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]
        
        for _ in range(n_hairs):
            hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(hair_images)))
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            # Creating a mask and inverse mask
            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Now black-out the area of hair in ROI
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Take only region of hair from hair image.
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            # Put hair in ROI and modify the target image
            dst = cv2.add(img_bg, hair_fg)

            img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst
                
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, hairs_folder="{self.hairs_folder}")'

class DrawHair:
    """
    Draw a random number of pseudo hairs

    Args:
        hairs (int): maximum number of hairs to draw
        width (tuple): possible width of the hair in pixels
    """

    def __init__(self, hairs:int = 4, width:tuple = (1, 2)):
        self.hairs = hairs
        self.width = width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL Image: Image with drawn hairs.
        """
        if not self.hairs:
            return img
        
        width, height, _ = img.shape
        
        for _ in range(random.randint(0, self.hairs)):
            # The origin point of the line will always be at the top half of the image
            origin = (random.randint(0, width), random.randint(0, height // 2))
            # The end of the line 
            end = (random.randint(0, width), random.randint(0, height))
            color = (0, 0, 0)  # color of the hair. Black.
            cv2.line(img, origin, end, color, random.randint(self.width[0], self.width[1]))
        
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, width={self.width})'

class Microscope:
    """
    Cutting out the edges around the center circle of the image
    Imitating a picture, taken through the microscope

    Args:
        p (float): probability of applying an augmentation
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to apply transformation to.

        Returns:
            PIL Image: Image with transformation.
        """
        if random.random() < self.p:
            circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8), # image placeholder
                        (img.shape[0]//2, img.shape[1]//2), # center point of circle
                        random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15), # radius
                        (0, 0, 0), # color
                        -1)

            mask = circle - 255
            img = np.multiply(img, mask)
        
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'

train_transform = transforms.Compose([
    AdvancedHairAugmentation(hairs_folder='melanoma-hairs'),
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    Microscope(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

DATA_FOLDER = "data"

def predictSingle(df):
    """ for single image """
    start = time.time()
    prediction = 0
    TTA = 3
    predict_df = df
    data_dict = torch.load("data/data_dict.pth")
    predict_data = MelanomaDataset(
        df=predict_df,
        imfolder="data", 
        train=False,
        transforms=train_transform,  # For TTA
        meta_features=data_dict["meta_features"]
    )

    predict_loader = DataLoader(dataset=predict_data, batch_size=1, shuffle=False, num_workers=0)
    preds = torch.zeros((len(predict_data), 1), dtype=torch.float32, device=device)

    for fold in range(5):
        model_path = f'model_{fold + 1}.pth'
        model = torch.load(os.path.join("model", model_path), map_location=torch.device('cpu'))
        # efficientnet_pytorch version 0.6 fixes attribute error
        model.eval()
        with torch.no_grad():
            tta_preds = torch.zeros((len(predict_data), 1), dtype=torch.float32, device=device)
            for _ in range(TTA):
                for i, x_test in enumerate(predict_loader):
                    x_test[0] = torch.tensor(x_test[0], device=device, dtype=torch.float32)
                    x_test[1] = torch.tensor(x_test[1], device=device, dtype=torch.float32)
                    z_test = model(x_test)
                    z_test = torch.sigmoid(z_test)
                    z_test *= 10
                    tta_preds[i*predict_loader.batch_size:i*predict_loader.batch_size + x_test[0].shape[0]] += z_test
            preds += tta_preds / TTA
    
    preds /= 5
    end = time.time()
    arr = np.array(preds)
    return arr[0][0]
    
st.title("Melanoma Classification")

st.header("Skin Cancer Melanoma Classification Example")

uploaded_person = st.file_uploader("Upload an image of the effected area", type="jpg")

age = st.slider("Enter patient age...", min_value=0, max_value=90)

age /= 90 # max value of ages during train time for normalizing

gender = st.sidebar.selectbox("specify the gender of patient...", ("male", "female"))

region = st.sidebar.selectbox(
    "Which region is patient affected ?",
    ("head/neck", "lower extremity", "oral/genital", "palms/soles", "torso", "upper extremity", "nan")
)

df = pd.read_csv("data/input.csv")
for col in df.columns:
    if col.startswith('site_'):
        df[col].iloc[0] = 0

df["site_" + str(region)].iloc[0] = 1
if str(gender) == "male":
    df["sex"] = 1
else:
    df["sex"] = 0

df["age_approx"] = age

if uploaded_person is not None:
    img = Image.open(uploaded_person)
    st.image(img, caption="User Input", width=None, use_column_width=False)
    img.save("data/img.jpg")


if st.button('Execute'):
    st.write("Classifying...")
    percent = predictSingle(df)
    percent *= 100
    execute_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.08)
        execute_bar.progress(percent_complete + 1)
    st.write('There is a ',percent,' percent chance that the person has Melanoma')



