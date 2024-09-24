# Detecting whether person is wearing mask or not.

## Setup conda enviroment 

```python
conda create -n MaskDetector python=3.10.0
conda activate MaskDetector
pip install -r requirements.txt
```

## Train model 

```python
python -i <IMAGE_PATH> -m train
```

Training model with config parameters gives 98.67% accuracy

## Test model 

```python
python -i <IMAGE_PATH> -m test
```

## Run application
```python 
python detect_in_real_time.py
```



Model has outputs 3 states - correctly, incorrectly, not wearing

OpenCV was used to detect faces.

Dataset for project
https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection



