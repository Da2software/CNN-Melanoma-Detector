# CNN-Melanoma-Detector

### Requirements

- *python 3.10.13*

### Create environment

> python -m venv venv

### Start the environment

```bash
# linux
source venv/bin/activate

# windows
venv\Scripts\activate
```

### Download dataset

Download Kaggle melanoma
dataset <a href="https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images?resource=download">
here</a>
<br>
Extract the main folder (melanoma_cancer_dataset folder).

### Train

```bash
python train_model.py <EPOCHS_NUMBER>
```

### Run API

```bash
fastapi dev main.py
```

### Endpoints

> POST
>
> body_type: form
>
> params: (image: file)
>
> NOTE: use images sizes around 224x224 to better accuracy, the endpoint
> rescales the image input to 224x224 but that can cause some precision issues.
>> http://127.0.0.1:8000/predict

 

> GET
>
> body_type: no need it
>
> params: no need it
>> http://127.0.0.1:8000/accuracy