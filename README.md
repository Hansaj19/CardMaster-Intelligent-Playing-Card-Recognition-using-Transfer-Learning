# Card Classification using VGG16 🃏🧠

A deep learning-based project to classify different playing cards using convolutional neural networks (CNN) and the VGG16 architecture.  
This is part of our academic project submission at VIT University.

---

## 📁 Project Structure


---

## 🚀 What's Inside?

| Folder | Contents |
|--------|----------|
| **Assignments/** | All 3 assignment PDFs |
| **Project Initialization.../** | Problem statement, planning reports |
| **Data Collection.../** | Data sources, quality check, preprocessing |
| **Model Development/** | Model selection, training Jupyter notebook |
| **Model Optimization.../** | Tuning report |
| **Project Executable Files/** | Flask app, trained model (`VGG16_model.h5`), test code |
| **Documentation & Demonstration/** | Final PDF and mp4 demo |

---

## 💻 How to Run the Project

1. Clone the repo:
```git clone <your-repo-link>```
2. Create virtual env
```console
python -m venv card_env
card_env\Scripts\activate
```
3. Install dependencies:
```pip install -r requirements.txt```
4. Run Flask App:
```console
cd "Project Executable Files\flask_app"
python app.py
```

### Tools & Tech Used
- Python 3.11+
- TensorFlow / Keras
- scikit-learn
- VGG16 pre-trained model
- Flask (for local deployment)
-  Jupyter Notebook

---
# Notes
Dataset are excluded from GitHub to keep repo clean.
### To download the dataset, run the following script in Notebook once.

```python
!mkdir -p ~/.kaggle
!copy kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d gpiosenka/cards-image-datasetclassification
!unzip cards-image-datasetclassification.zip -d cards_dataset
import zipfile
with zipfile.ZipFile("cards-image-datasetclassification.zip", 'r') as zip_ref:
    zip_ref.extractall("cards_dataset")
```

Make sure to place dataset manually or download as instructed above.
