## (1) LabelImg (로컬 설치형, 간단함) : 라벨링 툴
## (2) Roboflow (웹 기반, 자동화 지원)
## (3) CVAT (웹 기반, 전문가용) : 오픈소스 웹 플랫폼, 팀 협업 및 대규모 프로젝트에 적합

from ultralytics import YOLO
from config_env import get_config, get_config_path
import os, io, cv2, random, shutil, argparse, yaml,uuid

CONFIG = get_config()
def train_yolo():
    global CONFIG
    # 모델 로드
    model = YOLO('yolov8n.pt')

    # 모델 학습
    model.train(
        data='D:/project/umssumss/yolo8/Goats.v5i.yolov8/data.yaml',
        epochs=20,
        imgsz=512,
        device=0,
        augment=False,    # 데이터 증강 활성화
        hsv_h=0.015,     # Hue 변화
        hsv_s=0.7,       # Saturation 변화
        hsv_v=0.4,       # Value 변화
        translate=0.1,   # 이미지 이동
        scale=0.5,       # 이미지 스케일링
        flipud=0.5,      # 상하 반전 확률
        fliplr=0.5,      # 좌우 반전 확률
        mosaic=1.0,      # Mosaic 증강 사용 여부
        mixup=0.5,       # Mixup 증강 사용 여부
        lr0=0.001,                 # 초기 학습률 (Adam은 0.001 권장)
        optimizer='Adam',          # 옵티마이저 변경
        weight_decay=0.0005,       # Adam에서도 과적합 방지용 L2 정규화
        patience=0,
        resume=True
    )

def train_yolo2():
    global CONFIG
    # 모델 로드
    #model = YOLO('yolov8n.pt')
    model = YOLO('D:/project/umssumss/yolo8/runs/goat_default_train19/weights/best.pt')  # 경로는 모델에 맞게 변경!
    # 모델 학습
    model.train(
        data='D:/project/umssumss/yolo8/Goats.v5i.yolov8/data.yaml',
        epochs=150,
        imgsz=832,
        device=0,
        augment=False,    # 데이터 증강 활성화
        lr0=0.001,                 # 초기 학습률 (Adam은 0.001 권장)
        optimizer='Adam',          # 옵티마이저 변경
        weight_decay=0.0005,       # Adam에서도 과적합 방지용 L2 정규화
        patience=10,               # 조기 종료 설정 (10번 동안 개선 없으면 종료)
        resume=True
    )

def train_yolo_default():
    global CONFIG
    # 모델 로드
    print ("training.....................now")
    #model = YOLO('yolov8n.pt')
    model_path = CONFIG["train"]["model_path"]
    data_info = CONFIG["data"]["train_root"]
    data_yaml = data_info+"/data.yaml"
    data_save = data_info+"/runs"
    batch_info = CONFIG["train"]["batch"]
    epoch_info = CONFIG["train"]["epoch"]

    #model = YOLO('D:/project/umssumss/yolo8/runs/goat_default_train19/weights/best.pt')  # 경로는 모델에 맞게 변경!
    model = YOLO(model_path)  # 경로는 모델에 맞게 변경!
    # 모델 학습
    model.train(
        #data='D:/project/umssumss/yolo8/Goats.v5i.yolov8/data.yaml',
        data=data_yaml,
        epochs=epoch_info,             # 학습 반복 횟수 (충분히 학습)
        batch=batch_info,               # 배치 크기 (VRAM에 맞게 조정)
        imgsz=640,              # 입력 이미지 크기
        optimizer="SGD",        # 일반적으로 SGD가 적합 (AdamW도 가능)
        lr0=0.01,               # 초기 학습률
        lrf=0.1,
        device=0,               # 최종 학습률 비율
        weight_decay=0.0005,    # L2 정규화 (Overfitting 방지)
        momentum=0.937,         # SGD 모멘텀
        dropout=0.1,            # Dropout 적용 (일반화 성능 향상)
        augment=True,           # 데이터 증강 활성화
        resume=True,
        #project='D:/project/umssumss/yolo8/runs',  # 저장될 폴더 경로
        project=data_save,  # 저장될 폴더 경로
        name='default_train',  # 하위 폴더 이름 (weights는 그 안에 생성됨)
    )
    metrics = model.val()
    print(metrics)


def train_model(data_id : str, job_id : str) :
    global CONFIG
    # 모델 로드
    #"train": {
    #    "model_path": "./models/yolov8n.pt",
    #    "entry": "python train_from_web.py",
    #    "epochs": 600,
    #    "batch": 24
    #},
    print (f"training.....................data_info : {data_id}  job_id : {job_id}")
    #model = YOLO('yolov8n.pt')
    model_path = CONFIG["train"]["model_path"]
    entry = CONFIG["train"]["entry"]
    epoch_info = CONFIG["train"]["epochs"]
    batch_info = CONFIG["train"]["batch"]

    data_info = CONFIG["data"]["train_root"]+f"/{data_id}"
    data_yaml = data_info+"/data.yaml"
    data_save = data_info+"/runs"

    print(f"1------------------>{data_yaml}")

    #model = YOLO('D:/project/umssumss/yolo8/runs/goat_default_train19/weights/best.pt')  # 경로는 모델에 맞게 변경!
    model = YOLO(model_path)  # 경로는 모델에 맞게 변경!

    print(f"2 model_path------------------>{model_path} \n 학습시작합니다..........................\n학습정보 환경파일 :{data_yaml}")
    # 모델 학습
    model.train(
        data=data_yaml,
        epochs=epoch_info,             # 학습 반복 횟수 (충분히 학습)
        batch=batch_info,               # 배치 크기 (VRAM에 맞게 조정)
        imgsz=640,              # 입력 이미지 크기
        optimizer="SGD",        # 일반적으로 SGD가 적합 (AdamW도 가능)
        lr0=0.01,               # 초기 학습률
        lrf=0.1,
        device=0,               # 최종 학습률 비율
        weight_decay=0.0005,    # L2 정규화 (Overfitting 방지)
        momentum=0.937,         # SGD 모멘텀
        dropout=0.1,            # Dropout 적용 (일반화 성능 향상)
        augment=True,           # 데이터 증강 활성화
        resume=False,
        #project='D:/project/umssumss/yolo8/runs',  # 저장될 폴더 경로
        project=data_save,  # 저장될 폴더 경로
        name='default_train',  # 하위 폴더 이름 (weights는 그 안에 생성됨)
    )
    metrics = model.val()
    print(metrics)


if __name__ == '__main__':

    ap = argparse.ArgumentParser()

    ap.add_argument("--data_info", type=str, default="")
    ap.add_argument("--job_id", type=str, help="jobid 입력" )
    args = ap.parse_args()
    print(f" train_from_web {args}")
    if(args.job_id == None) :
        print(" no parameter")
        exit(1)
    else :
        print(args.job_id)
        #label_ext(args.target)
        job_id = args.job_id
        data_info = args.data_info
        train_model(data_info, job_id)


