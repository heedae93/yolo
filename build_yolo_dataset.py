import os, io, cv2, random, shutil, argparse, yaml,uuid
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import cv2, numpy as np
from sqlalchemy import create_engine, text
import psycopg2
from psycopg2.extras import RealDictCursor
import logging, psycopg2
from psycopg2.extras import execute_batch
from config_env import CONFIG_PATH
from config_env import get_config, get_config_path


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
            "chicken", "guard dog", "goat","puppy", "cat", "crocodile", "tiger"]

# ===== Postgres 연결 =====

JOB_ID=-1

def init_db():

    print("init database...............................")
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


    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS job (
        job_id       TEXT        NOT NULL,
        file_name   TEXT        NOT NULL,
        ts TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_job_id ON frames(job_id );")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS frames (
        id           BIGSERIAL PRIMARY KEY,           -- 자동증가 PK
        job_id       TEXT        NOT NULL,
        frame_index  INTEGER     NOT NULL,
        image_path   TEXT        NOT NULL,
        width        INTEGER     NOT NULL,
        height       INTEGER     NOT NULL,
        ts TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_frames_job ON frames(job_id, id);")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS boxes (
        id BIGSERIAL PRIMARY KEY,
        frame_id INTEGER,
        cls TEXT,
        conf REAL,
        x1 numeric, y1 numeric, x2 numeric, y2 numeric
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_boxes_frame ON boxes(frame_id);")
    conn.commit()
    conn.close()

init_db()

def to_int(v):
    # numpy scalar / torch tensor / list 한 요소 등을 안전하게 스칼라 int로
    if hasattr(v, "item"):           # numpy, torch scalar
        v = v.item()
    v = float(np.asarray(v).reshape(()))  # 배열이면 0차원 스칼라로
    return int(round(v))

def extract_frames(video_path: Path, out_dir: Path, every_n=20, max_frames=None):
    print("extract_frames...............................")
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] cannot open: {video_path}")
        return []
    frames = []
    idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % every_n == 0:
            save_path = out_dir / f"{video_path.stem}_{idx:06d}.jpg"
            cv2.imwrite(str(save_path), frame)
            frames.append(save_path)
            saved += 1
            if max_frames and saved >= max_frames:
                break
        idx += 1
        #print(f"{idx}")
    cap.release()
    return frames

def coco_names(model):
    # ultralytics YOLO 모델의 클래스명 dict 반환
    return model.names  # {id: name}

def save_yolo_label(label_path: Path, boxes, img_w, img_h):
    #print("save_yolo_label...............................")
    """
    boxes: [(cls_id, x1,y1,x2,y2, conf)]
    YOLO txt: <cls> cx cy w h  (모두 0~1 정규화)
    """
    lines = []
    for cls, x1, y1, x2, y2, conf in boxes:
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines))
    #print(f"{lines}")

def save_img_bounding(label_path: Path, boxes, img_w, img_h, img_path, job_id, frame_id):
    #print("save_img_bounding...............................")
    img = cv2.imread(str(img_path))
    if img is None:
        return

    conn = getCon()

    payload = []
    idx =0
    for cls, x1, y1, x2, y2, conf in boxes:
        cx = ((x1 + x2) / 2)
        cy = ((y1 + y2) / 2)
        w  = (x2 - x1)
        h  = (y2 - y1)
        cx = to_int(x1)
        cy = to_int(y1)
        cx2 = to_int(x2) #+ to_int(w)
        cy2 = to_int(y2) #+ to_int(h)
        # 박스 그리기 (파란색, 두께 2)
        cv2.rectangle(img, (cx, cy), (cx2, cy2), (255, 0, 0), 2)
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
        payload.append({
            "fid": frame_id,
            "cls": cls,
            "conf": float(conf) if conf is not None else None,
            "x1": cx, "y1": cy, "x2": cx2, "y2": cy2
        })
        idx = idx + 1

    #print(f"job_id --> {job_id} {idx} {label_path} {img_w} {img_h}")
        # 텍스트 추가
    if boxes:
        try:
            with conn:
                with conn.cursor() as cur:

                    fr = cur.execute(
                        """
                            INSERT INTO frames(job_id, frame_index, image_path, width, height)
                            VALUES (%s, %s, %s, %s, %s)
                            returning id
                        """,
                        ( job_id,  frame_id,  str(label_path), img_w,  img_h )
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
        finally:

            conn.close()


    label_path.parent.mkdir(parents=True, exist_ok=True)
    #print(f"{label_path}")
    cv2.imwrite(label_path, img)




def auto_label_images(model, image_paths, labels_dir, classes_keep=None, conf_thres=0.35, iou=0.5, job_id=-1):
    #print("auto_label_images...............................")
    id2name = coco_names(model)  # {0:'person', ...}
    name2id = {v: k for k, v in id2name.items()}

    if classes_keep:
        invalid = [c for c in classes_keep if c not in name2id]
        if invalid:
            print(f"[WARN] unknown class names: {invalid}")
        keep_ids = {name2id[c] for c in classes_keep if c in name2id}
    else:
        keep_ids = set(id2name.keys())
    frame_id = 0
    for img_path in tqdm(image_paths, desc="Auto-label"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] bad image: {img_path}")
            continue
        h, w = img.shape[:2]

        #print(f"h   {h} w   {w}")
        # 예측
        res = model.predict(source=img, conf=conf_thres, iou=iou, verbose=False)
        boxes_out = []
        for r in res:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls[0])
                if cls_id not in keep_ids:
                    continue
                xyxy = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
                conf = float(b.conf[0])
                x1,y1,x2,y2 = map(float, xyxy)
                # 이미지 경계 클리핑
                x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
                y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes_out.append((cls_id, x1,y1,x2,y2, conf ))

        # 라벨 저장
        label_path = labels_dir / (img_path.stem + ".txt")
        if boxes_out:
            save_yolo_label(label_path, boxes_out, w, h)
            label_path2 = labels_dir / (img_path.stem + ".jpg")
            save_img_bounding(label_path2, boxes_out, w, h,img_path, job_id, frame_id)
        else:
            # 객체가 없으면 빈 파일(선택사항: 생성하지 않음)
            label_path.write_text("")
        frame_id = frame_id +1


def split_dataset(images_all, out_root, train=0.8, val=0.2, test=0.0, seed=42):
    random.seed(seed)
    images = list(images_all)
    random.shuffle(images)
    n = len(images)
    n_train = int(n*train)
    n_val   = int(n*val)
    train_set = images[:n_train]
    val_set   = images[n_train:n_train+n_val]
    test_set  = images[n_train+n_val:]
    # 이동
    for subset, name in [(train_set,"train"), (val_set,"val"), (test_set,"test")]:
        for img in subset:
            lbl = (out_root/"labels"/"all"/(img.stem+".txt"))
            dst_img = out_root/"images"/name/img.name
            dst_lbl = out_root/"labels"/name/(img.stem+".txt")
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(img), str(dst_img))
            if lbl.exists():
                shutil.move(str(lbl), str(dst_lbl))
    # 중간 all 폴더 정리
    #shutil.rmtree(out_root/"labels"/"all", ignore_errors=True)

def write_data_yaml(yaml_path, class_dict, dataset_root):
    data = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "names": [class_dict[i] for i in sorted(class_dict.keys())]
    }
    yaml_path.write_text(yaml.dump(data, sort_keys=False, allow_unicode=True))

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=str, required=True, help="동영상 폴더 또는 단일 파일")
    ap.add_argument("--out", type=str, default="./static/yolo_dataset")
    ap.add_argument("--every", type=int, default=5, help="N프레임마다 1장 추출")
    ap.add_argument("--max-per-video", type=int, default=None, help="영상별 최대 프레임 수")
    ap.add_argument("--model", type=str, default="yolov8n.pt")
    ap.add_argument("--classes", type=str, default="", help="쉼표로 구분된 클래스명만 유지 (예: person,car,bottle)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--jobid", type=str, default= uuid.uuid4().hex )

    args = ap.parse_args()
    print(f"{args}")
    videos_arg = Path(args.videos)
    out_root = Path(args.out +"/"+args.jobid)
    #out_root = Path(args.out )

    insert_job(args.videos, args.jobid)

    imgs_all_dir = out_root / "images" / "all"
    lbls_all_dir = out_root / "labels" / "all"
    imgs_all_dir.mkdir(parents=True, exist_ok=True)
    lbls_all_dir.mkdir(parents=True, exist_ok=True)

    # 1) 비디오 수집
    video_files = []
    if videos_arg.is_file():
        video_files = [videos_arg]
    else:
        for ext in ("*.mp4","*.mov","*.avi","*.mkv"):
            video_files += list(videos_arg.rglob(ext))
    if not video_files:
        print("[ERR] no video files found.")
        return

    # 2) 프레임 추출
    all_images = []
    for v in video_files:
        frames = extract_frames(v, imgs_all_dir, every_n=args.every, max_frames=args.max_per_video)
        all_images += frames
    print(f"[INFO] extracted frames: {len(all_images)}")

    # 3) 모델 로드 + 자동 라벨
    model = YOLO(args.model)
    keep = [s.strip() for s in args.classes.split(",") if s.strip()] if args.classes else None
    JOB_ID = args.jobid
    print(f"job_id --->    {JOB_ID}")
    auto_label_images(model, all_images, lbls_all_dir, classes_keep=keep, conf_thres=args.conf, iou=args.iou, job_id = args.jobid)

    # 4) 데이터 분할(train/val/test)
    split_dataset(all_images, out_root, train=0.8, val=0.2, test=0.0)

    # 5) data.yaml 생성
    class_dict = coco_names(model)  # {id: name}
    write_data_yaml(out_root/"data.yaml", class_dict, out_root)

    print("\n[DONE]")
    print(out_root)
    print(" - images/train, images/val")
    print(" - labels/train, labels/val")
    print(" - data.yaml")
    print("\n학습 예:")
    print(f"yolo detect train data={str((out_root/'data.yaml').resolve())} model={args.model} imgsz=640 epochs=50 batch=16")
    print("\n주의: 자동 라벨은 오차가 있으니, 꼭 도구로 교정하세요.")

if __name__ == "__main__":
    main()
