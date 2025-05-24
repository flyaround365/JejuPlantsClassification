
import torchmetrics

# 데이터 불러오는 함수
import os

import cv2          # 패키지 없음. Terminal에서 pip install opencv-python
import math


def get_jabaebong_data(data_dir, width=224, height=224):
    classes = os.listdir(data_dir)
    classes = [c for c in classes if os.path.isdir(os.path.join(data_dir, c))]
    assert len(classes) > 0
    data = {c: list() for c in classes}

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for file in os.listdir(cls_dir):
            image = cv2.imread(os.path.join(cls_dir, file))
            if image is None:
                print("[W] Failed to load image: {}".format(file))
                continue

            img_h, img_w = image.shape[:2]
            if (img_h / height) > (img_w / width):
                new_w = width
                new_h = math.ceil(img_h / img_w * new_w)
            else:
                new_h = height
                new_w = math.ceil(img_w / img_h * new_h)

            image = cv2.resize(image, (new_w, new_h))
            if new_w > new_h:
                diff = new_w - new_h
                left = diff // 2
                right = diff - left
                image = image[:, left:-right]
            elif new_h > new_w:
                diff = new_h - new_w
                left = diff // 2
                right = diff - left
                image = image[left:-right, :]

            data[cls].append(image)

    return data

# 데이터 불러오기
data_names = [
    "bonggae",
    "dosim",
    "gimnyeong",
    "jabaebong",
    "jogeun",
    "jongdal",
]
data_name = data_names[0]
data_dir = f"/mnt/raid/jwa/data/image/{data_name}"
data = get_jabaebong_data(data_dir, 512, 512)
print(f"Loaded data with {len(data)} classes.")

# 모델 코드
import gc
import multiprocessing
import os
import random
import re
import traceback

import torch
from torchmetrics.functional.classification import multiclass_accuracy
from tqdm import tqdm


def preprocess_blip_input_text(text, max_words=50):
    text = re.sub(
        r"([.!\"()*#:;~])",
        " ",
        text.lower(),
    )
    text = re.sub(
        r"\s{2,}",
        " ",
        text,
    )
    text = text.rstrip("\n")
    text = text.strip(" ")

    # truncate caption
    caption_words = text.split(" ")
    if len(caption_words) > max_words:
        text = " ".join(caption_words[: max_words])

    return text


def get_scores_zero_shot(data, model_str, method_str, result_q=None):
    from transformers import AutoProcessor, AutoModel

    model = AutoModel.from_pretrained(model_str)
    processor = AutoProcessor.from_pretrained(model_str)

    model = model.cuda()  # gpu 사용할 때 쓰는 코드

    idx2cls = list(sorted(data.keys()))
    cls2idx = {c:i for i, c in enumerate(idx2cls)}

    img_and_label = []
    for cls, images in data.items():
        img_and_label.extend([(img, cls) for img in images])

    classes_ko = []
    classes_en = []
    for cls in idx2cls:
        idx = cls.index('(')
        ko = cls[:idx-1]
        en = cls[idx+1:-1]
        classes_ko.append(ko)
        classes_en.append(en)

    label_ids = [cls2idx[cls] for img, cls in img_and_label]
    label_ids = torch.tensor(label_ids)

    if method_str == "en":
        classes = classes_en
    elif method_str == "ko":
        classes = classes_ko
    elif method_str == "en_ko":
        classes = [f"{e} ({k})" for e, k in zip(classes_en, classes_ko)]
    elif method_str == "en_ko_prompt":
        classes = [f"{e} (known as '{k}' in Korean)." for e, k in zip(classes_en, classes_ko)]
    elif method_str == "en_prompt":
        classes = [f"This is a photo of '{c}'." for c in classes_en]
    else:
        raise NotImplementedError(method_str)

    inputs_kwargs = {
        "text": classes,
        # "images": [img for img, cls in img_and_label],
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
    }

    if model_str.find("google/siglip-") >= 0:
        inputs_kwargs["padding"] = "max_length"
    elif model_str.find("google/siglip2-") >= 0:
        inputs_kwargs["padding"] = "max_length"
        inputs_kwargs["max_length"] = 64
    elif model_str.find("Salesforce/blip-") >= 0:
        inputs_kwargs["text"] = [preprocess_blip_input_text(t) for t in inputs_kwargs["text"]]

    print(inputs_kwargs)
    inputs_kwargs["images"] = [img for img, cls in img_and_label]

    inputs = processor(**inputs_kwargs)
    all_img_inputs = inputs.data["pixel_values"]
    predictions = []
    batch_size = 1
    with tqdm(total=all_img_inputs.shape[0]) as pbar:
        for start in range(0, all_img_inputs.shape[0], batch_size):
            end = min(start + batch_size, all_img_inputs.shape[0])
            inputs.data["pixel_values"] = all_img_inputs[start:end]
            inputs = inputs.to(model.device)

            outputs = model(**inputs)
            pred = outputs.logits_per_image.argmax(dim=-1)
            predictions.append(pred)

            pbar.update(batch_size)

    predictions = torch.cat(predictions, dim=0)
    predictions = predictions.cpu()

    macro_accuracy = multiclass_accuracy(predictions, label_ids, len(idx2cls), average='macro')
    micro_accuracy = multiclass_accuracy(predictions, label_ids, len(idx2cls), average='micro')
    per_class_accuracy = multiclass_accuracy(predictions, label_ids, len(idx2cls), average='none')

    scores = {
        "macro": macro_accuracy.cpu().item(),
        "micro": micro_accuracy.cpu().item(),
        "per_class": per_class_accuracy.cpu().tolist(),
    }

    if result_q is not None:
        result_q.put(scores)

    return scores

exp_names = [
    "en",
    "ko",
    "en_ko",
    "en_ko_prompt",
    "en_prompt",
]

exp_name = exp_names[0]
# data_dir = f"/Users/jwa/Desktop/{data_name}"          # 데이터 위치
# output_dir = f"/content/out_dir/esac/{exp_name}"      # 원본 위치
output_dir = f"/mnt/raid/jwa/out/esac/{exp_name}"        # 수정 위치
os.makedirs(output_dir, exist_ok=True)

small_model_list = [
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-large-patch14-336",

    "google/siglip-so400m-patch14-224",
    "google/siglip-base-patch16-256",
    "google/siglip-base-patch16-384",
    "google/siglip-base-patch16-512",
    "google/siglip-base-patch16-256-multilingual",
    "google/siglip-base-patch16-224",

    "google/siglip2-base-patch16-512",
    "google/siglip2-base-patch16-384",
    "google/siglip2-base-patch16-256",
    "google/siglip2-base-patch16-224",
    "google/siglip2-base-patch32-256",
]

model_str = small_model_list[4]
print(f"Running {model_str}")
scores = get_scores_zero_shot(data, model_str, exp_name)

print(f"Macro accuracy: {scores['macro']:.4f}, micro accuracy: {scores['micro']:.4f}")

# confusion matrix
import matplotlib.pyplot as plt

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

fig, ax = scores["conf_mat"].plot()
set_size(7, 7, ax)

plt.show()
idx2cls = list(sorted(data.keys()))
for i, cls in enumerate(idx2cls):
    print(f"{i}: {cls}")