import os, uuid, cv2, random, shutil, argparse
from ultralytics import YOLO
from PIL import Image
import psycopg2
from psycopg2.extras import RealDictCursor
import logging, psycopg2
from psycopg2.extras import execute_batch
from sqlalchemy import create_engine, text
import cv2
from pathlib import Path
import numpy as np
import torch
from config_env import get_config, get_config_path

CONFIG = get_config()

class_name =["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
            "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
            "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush",
            "chicken", "guard dog", "goat","puppy", "cat", "crocodile", "tiger", "foodbin"]

# YOLOv8 모델 로드 (사전 학습된 모델 사용)


#model = YOLO('yolov8n.pt')
model_path = CONFIG["train"]["model_path"]
data_info = CONFIG["data"]["train_root"]
data_save = data_info+"/runs"


#model = YOLO("yolov8n.pt")
#model = YOLO('D:/project/umssumss/yolo8/dataset/foodbin/runs/goat_default_train6/weights/best.pt')  # 경로는 모델에 맞게 변경!
model = YOLO(model_path)  # 경로는 모델에 맞게 변경!

def getCon() :
    config = get_config()
    #print(f"{config}")
    dbinfo = config["database"]
    conn = psycopg2.connect(
        dbname=dbinfo["dbname"],          # ✅ DB 이름
        user=dbinfo["user_id"],
        password=dbinfo["password"],
        host=dbinfo["host"],
        port=dbinfo["port"],
        # sslmode="require",    # 필요시
    )
    return conn

"""
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://user:pass@192.168.0.231:5432/public"
)
engine = create_engine(
    DATABASE_URL,
    pool_size=10, max_overflow=20,
    pool_pre_ping=True, future=True
)
"""

def insert_job(vides_arg, jobid) :
    conn = getCon()
    try:
        with conn:
            with conn.cursor() as cur:

                cur.execute(
                    """
                        INSERT INTO job(job_id, file_name)
                        VALUES (%s, %s)
                        returning job_id
                    """,
                    ( jobid,  vides_arg)
                )
                print(f"inserted ")
    except psycopg2.Error as e:
        logging.error("PG error: %s", e)
    finally:
        conn.close()

def to_int(v):
    """
    torch.Tensor / numpy scalar / python scalar / list 한 요소 등을
    안전하게 int로 변환
    """
    if isinstance(v, torch.Tensor):
        # torch scalar tensor
        if v.numel() == 1:
            return int(round(v.item()))
        else:
            # multi-element tensor → numpy 변환 후 첫 값 사용
            v = v.detach().cpu().numpy().reshape(-1)[0]
            return int(round(float(v)))

    if hasattr(v, "item"):  # numpy scalar 같은 경우
        v = v.item()

    if isinstance(v, (list, tuple)):
        v = v[0]

    return int(round(float(v)))


def save_img_bounding(label_path: Path, db_save_path, boxes, img_w, img_h, img_path, job_id, frame_id):
    #print("save_img_bounding...............................")
    img = cv2.imread(str(img_path))
    if img is None:
        return

    conn = getCon()
    payload = []
    idx =0

    for sublist in boxes:  #박스정보는 화면에 그리는 정보임
        for cls, x1, y1, x2, y2, conf in sublist:
            print(f"sublist ----------------->{x1} {y1} {x2} {y2} " )
            cx = ((x1 + x2) / 2)
            cy = ((y1 + y2) / 2)
            w  = (x2 - x1)
            h  = (y2 - y1)
            cx = to_int(x1)
            cy = to_int(y1)
            cx2 = to_int(x2)  #+ to_int(w)
            cy2 = to_int(y2) #+ to_int(h)
            # 박스 그리기 (파란색, 두께 2)
            """
            cv2.rectangle(img, (cx*img_w, cy*img_h), (cx2*img_w, cy2*img_h), (255, 0, 0), 2)
            cv2.putText(
                img,
                f"{class_name[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}",
                (int(x1), int(y1) + 15),          # 글자 기준점(왼쪽-아래). 박스 위로 15px 아래에 출력
                cv2.FONT_HERSHEY_SIMPLEX,        # 폰트
                0.5,                              # 폰트 크기
                (0, 0, 255),                      # 색(BGR) - 빨강
                1,                                # 두께
                cv2.LINE_AA                       # 안티앨리어싱
            )
            """
            payload.append({
                "fid": frame_id,
                "cls": cls,
                "conf": float(conf) if conf is not None else None,
                "x1": cx, "y1": cy, "x2": cx2, "y2": cy2
            })
            idx = idx + 1

    cv2.imwrite(label_path, img)

    print(f"boxes   {boxes}")

    try:
        with conn:
            with conn.cursor() as cur:

                fr = cur.execute(
                    """
                        INSERT INTO frames(job_id, frame_index, image_path, width, height)
                        VALUES (%s, %s, %s, %s, %s)
                        returning id
                    """,
                    ( job_id,  frame_id,  str(db_save_path), img_w,  img_h )
                )
                frame_id = cur.fetchone()
                #print(f"frame_id {frame_id}")
                rows = []
                sql = """
                    INSERT INTO boxes(frame_id, cls, conf, x1, y1, x2, y2)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    returning id
                """
                for row in payload:
                    rows.append((
                        frame_id,
                        str(row["cls"]),
                        None if row.get("conf") is None else float(row["conf"]),
                        float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
                    ))

                ret = cur.executemany(sql, rows)
                #print(f"inserted ")
    except psycopg2.Error as e:
        logging.error("PG error: %s", e)

    conn.close()

def label_ext(tmp_dir) :
    job_id = uuid.uuid4().hex
    insert_job("yolo_dataset", job_id)
    # 이미지가 있는 디렉토리 설정 (필요한 경로로 변경)
    #tmp_dir="foodbin"
    #tmp_dir="download"

    model_path = CONFIG["train"]["model_path"]
    data_info = CONFIG["data"]["train_root"]
    data_save = data_info+"/runs"

    """"
    "collect_root": "/data/ingest",
        "train_root": "D:/project/umssumss/yolo8/avi_extracter",
        "image_dir": "D:/project/umssumss/yolo8/dataset",
        "label_dir": "D:/project/umssumss/yolo8/dataset",
        "web_root": "D:/project/umssumss/yolo8/avi_extracter"
    """
    yml_src = "./data.yaml"
    yml_dir  =  CONFIG["data"]["image_dir"]+ f"/{tmp_dir}/"  # 이미지가 저장된 폴더 경로
    image_dir =  CONFIG["data"]["image_dir"]+ f"/{tmp_dir}/images"  # 이미지가 저장된 폴더 경로
    label_dir =  CONFIG["data"]["label_dir"]+ f"/{tmp_dir}/labels"  # YOLO 라벨 저장 폴더
    all_image_dir =  CONFIG["data"]["web_root"]+ f"/static/yolo_dataset/{job_id}/{tmp_dir}_all"  # YOLO 라벨 저장 폴더
    save_image_dir = f"static/yolo_dataset/{job_id}/{tmp_dir}_all"  # YOLO 라벨 저장 폴더



    # 라벨 폴더 없으면 생성
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(all_image_dir, exist_ok=True)

    # 지원되는 이미지 확장자
    valid_extensions = (".jpg", ".jpeg", ".png", ".jfif")

    # 디렉토리 내 모든 이미지 파일을 가져오기
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]



    frame_id = 0
    # YOLO 형식으로 변환 및 저장
    print(f"파일 갯수 ----> {len(image_files)}")
    for image_file in image_files:
        print(f"frame id --->{frame_id}")
        img_path = None
        try :
            img_path = os.path.join(image_dir, image_file)
        except Exception as e:
            logging.error("PG error: %s", e)
            continue

        print(f"파일 이름은 {img_path}")
        # 이미지 크기 확인
        img = Image.open(img_path)
        print("open image")
        img_width, img_height = img.size

        # YOLO 예측 수행
        results = model.predict(img_path, conf=0.5)

        # YOLO 형식 라벨 저장을 위한 리스트
        #input(f"결과 {results}")
        yolo_labels = []
        boxes = []
        if(results == None ) :

            x_center = img_width / 2
            y_center = img_height / 2
            width =  img_width / 4
            height = img_height / 4
            tmp = [( 87,x_center - width/2,  y_center - height/2,
                        x_center + width/2, y_center + height/2,  0.0)]
            boxes.append(tmp) # 2개를 추가한다.
            tmp = [( 87,  x_center - width/2+5,  y_center - height/2+5,
                        x_center + width/2 +5,  y_center + height/2+5,  0.0)]
            boxes.append(tmp)
            print(f"no results-->")
        else :
            cls = 87
            conf = 0
            for result in results:
                for box in result.boxes:
                    #print(f"box    {box}")
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    print(f"class_id   {cls} confidence {conf}")
                    x1, y1, x2, y2 = box.xyxy[0]  # 바운딩 박스 좌표
                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    # 화면에 그리기 위한 정보는 그대로 넣음
                    tmp = [( cls,  x1,  y1,
                        x2 ,  y2,  conf)]
                    boxes.append(tmp)


                    # YOLO 라벨 형식 (클래스 ID는 0으로 설정, 필요하면 수정)
                    yolo_label = f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    yolo_labels.append(yolo_label)
                    print(yolo_label)

        # 라벨 저장 파일명 설정
        label_filename = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_filename)

        frame_id =  frame_id + 1
        # YOLO 라벨을 txt 파일로 저장
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_labels))

        print(f"라벨 저장 완료: {label_path}")

        #이미지에 박스포함 저장

        file_name = os.path.splitext(image_file)[0]
        label_path2 = all_image_dir+"/"+ (file_name + ".jpg")
        db_save_path = save_image_dir+"/"+ (file_name + ".jpg")
        print(f"저장이미지 : {label_path2}")
        save_img_bounding(label_path2, db_save_path, boxes, img_width, img_height, img_path, job_id, frame_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, required=False, help="타겟정보")
    args = ap.parse_args()

    if(args.target == None) :
        label_ext("foodBin")
    else :
        print(args.target)
        #label_ext(args.target)

if __name__ == '__main__':
    #train_yolo_default()
    main()



