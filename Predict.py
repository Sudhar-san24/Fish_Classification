# ============================
#  🐟 FISH SPECIES PREDICTION
# ============================

import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ============================
# 1️⃣ CREATE CLASS LABELS JSON (if not exists)
# ============================

dataset_train_dir = r'D:\Fish_Detection\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\train'
json_path = r'D:\Fish_Detection\Dataset\class_labels.json'

if not os.path.exists(json_path):
    print("⚠️  class_labels.json not found — creating automatically...")
    classes = sorted(os.listdir(dataset_train_dir))
    labels = {str(i): cls for i, cls in enumerate(classes)}

    with open(json_path, 'w') as f:
        json.dump(labels, f, indent=4)
    print(f"✅ class_labels.json created at: {json_path}")
else:
    print("✅ class_labels.json found.")


# ============================
# 2️⃣ LOAD LABELS
# ============================

with open(json_path, 'r') as f:
    labels = json.load(f)


# ============================
# 3️⃣ DEFINE PREDICTION FUNCTION
# ============================

def Predict_Fish(img, model_path):
    print(f"\n🔍 Predicting with model: {os.path.basename(model_path)}")

    try:
        model = load_model(model_path)
        pred = model.predict(img)
        pred_class = np.argmax(pred)
        predicted_fish = labels[str(pred_class)]
        print(f"🎯 Predicted Fish Type: {predicted_fish}")
    except Exception as e:
        print(f"❌ Error while predicting with {model_path}: {e}")


# ============================
# 4️⃣ MODEL PATHS
# ============================

custom_cnn_path = r'D:\Fish_Detection\MainApp\Custom CNN\custom_cnn_model.h5'
vgg_path = r'D:\Fish_Detection\MainApp\VGG16\vgg_finetuned_model.h5'
resnet_path = r'D:\Fish_Detection\MainApp\ResNet50\resnet_finetuned_model.h5'
mobilenet_path = r'D:\Fish_Detection\MainApp\MobileNet\mobilenet_finetuned_model.h5'
inception_path = r'D:\Fish_Detection\MainApp\InceptionV3\inception_finetuned_model.h5'
efficientnet_path = r'D:\Fish_Detection\MainApp\EfficientNetB0\efficientnet_finetuned_model.h5'


# ============================
# 5️⃣ TEST IMAGE PREPROCESSING
# ============================

test_img_path = r'D:\Fish_Detection\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\test\fish sea_food black_sea_sprat\0I5O9H5AFIAE.jpg'

if not os.path.exists(test_img_path):
    raise FileNotFoundError(f"❌ Test image not found at {test_img_path}")

img = image.load_img(test_img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0


# ============================
# 6️⃣ RUN PREDICTIONS FOR ALL MODELS
# ============================

model_paths = [
    custom_cnn_path,
    vgg_path,
    resnet_path,
    mobilenet_path,
    inception_path,
    efficientnet_path
]

for model_path in model_paths:
    if os.path.exists(model_path):
        Predict_Fish(img_array, model_path)
    else:
        print(f"⚠️ Model not found: {model_path}")

print("\n✅ All predictions completed successfully!")
