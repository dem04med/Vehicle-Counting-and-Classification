# Vehicle-Counting-and-Classification
Counting and Classifying Vehicles in Video Using YOLOv5

# Objective
Create an automatic system that detects and counts vehicles by type in traffic videos, using YOLOv5, with evaluation of its accuracy and performance.

## Environment Setup

```bash
python -m venv venv

./venv/Scripts/activate    # Actiavtio on Windows
source venv/bin/activate   # Activation on macOS/Linux
```
## Dependencies
```bash
git clone https://github.com/nicolai256/deep_sort.git deep_sort
git clone https://github.com/ultralytics/yolov5.git yolov5
pip install -r requirements.txt
```

## OpenRouterAPI_KEY
Aceder o seguinte link: https://openrouter.ai/keys
Efetuar login e, posteriormente, criar API_KEY.
```bash
mkdir .env
```
No seu ficheiro .env definir a variável correspondente à API_KEY criada.
```bash
OPENROUTER_API_KEY=SUA_API_KEY...
```

## Data Model Training
Considering that the folder referring to the cloned YOLO repository has the name yolov5:
```bash
cd .\Vehicle-Counting-and-Classification\yolov5
python train.py --img 640 --batch 4 --epochs 20 --data C:\YOLO\Vehicle-Counting-and-Classification\data\VeiculoT2\data.yaml --weights yolov5s.pt --cache --workers 0
```
After training, results are generated in yolov5/runs/train/weights (best.pt and last.pt). To check which of the two files performs better, run the following commands:
```bash
python val.py --weights runs/train/exp2/weights/best.pt --data C:\YOLO\Vehicle-Counting-and-Classification\data\VeiculoT2\data.yaml --img 640   #best.pt
python val.py --weights runs/train/exp2/weights/last.pt --data C:\YOLO\Vehicle-Counting-and-Classification\data\VeiculoT2\data.yaml --img 640   #last.pt
```

## Input Videos
Assuming you are in the Vehicle-Counting-and-Classification directory
```bash
mkdir input_videos
```
In the created directory place videos filmed or from other sources (e.g. Youtube)
```bash
yt-dlp -f best -o "video.mp4" "https://www.youtube.com/watch?v=MNn9qKG2UFI&list=PLcQZGj9lFR7y5WikozDSrdk6UCtAnM9mB&index=1&pp=iAQB"  #video from Youtube
```

## How to Run
First, make sure you have the output directory:
```bash
mkdir output
```
Next, in the main.py file, set the most appropriate value for the model.conf variable (it may need to be changed depending on the video used).
Then run the following commands:
```bash
cd scripts
python main.py
```
After executing main.py, the video will be generated with the detection and counting of each type of vehicle identified, as well as a .txt file with details about each class detected in the video.
