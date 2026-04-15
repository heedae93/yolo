import os, uuid, threading, subprocess, io
from pathlib import Path
from typing import Dict, Any
from flask import Flask, request, jsonify, send_file, render_template, abort
from werkzeug.utils import secure_filename
import cv2
from sqlalchemy import create_engine, text
import psycopg2
from psycopg2.extras import RealDictCursor
import logging, psycopg2
from psycopg2.extras import execute_batch
import json

from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from PIL import Image
import shutil
from config_env import get_config, get_config_path

import zipfile
from datetime import datetime

# ====== 설정 ======

config = get_config()
BASE_DIR    = Path(__file__).parent.resolve()

UPLOAD_DIR  = BASE_DIR / "uploads"
LOG_DIR     = BASE_DIR / "logs"
FRAME_DIR  = BASE_DIR / "frames"

limit = 10

UPLOAD_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# 실행할 외부 파이썬 프로그램

ALLOWED_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

EXTERNAL_CMD = ["python", str(BASE_DIR / "build_yolo_dataset.py")]
EXTERNAL_CMD2 = ["python", str(BASE_DIR / "label_ext.py")]
EXTERNAL_CMD3 = ["python", str(BASE_DIR / "train_from_web.py")]

# ===== Postgres 연결 =====
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"postgresql+psycopg2://user:pass@{config['database']['host']}:{config['database']['port']}/public"
)
engine = create_engine(
    DATABASE_URL,
    pool_size=10, max_overflow=20,
    pool_pre_ping=True, future=True
)

# ====== 간단 Job Store ======
JOBS = {}  # job_id -> dict(status, file_path, log_path, pid, returncode)
JOBS_LOCK = threading.Lock()
MODEL = None
CONFIG = None

UPLOAD_DIR = "./uploads"
EXTRACT_DIR = "./extracted"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)



def new_job(file_path: Path) -> str:
    job_id = uuid.uuid4().hex
    log_path =  f"{LOG_DIR }/{job_id}.log"
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"[INFO] Job {job_id} created for {file_path}\n")
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "file_path": str(file_path),
            "log_path": str(log_path),
            "pid": None,
            "returncode": None,
        }
    return job_id

def stop_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        print(f"종료하는 잡은 {job}  입니다.")
        if job and ( job.get("status") =='running' ) :
            print(f"....................................{job}")
            proc = job["process"]
            proc.terminate()  # SIGTERM
            proc.kill()     # SIGKILL
            job["status"] = "stopping"
            print(f"종료합니다. {job}")
            return True
    return False



def run_job_async(job_id: str, param : str, data_info : str):

    print(f"*************************************   {job_id}   {param}")

    with JOBS_LOCK:
        meta = JOBS.get(job_id)
    if not meta:
        return
    file_path = meta["file_path"]
    log_path  = meta["log_path"]

    log_fh = open(log_path, "a", encoding="utf-8")
    log_fh.write("[INFO] Starting process...\n"); log_fh.flush()
    print(f" starting process..........")
    try:
        if param =='1' : # 동영상 추출
            cmd = EXTERNAL_CMD
        elif param =='2': #라벨 추출
            cmd = EXTERNAL_CMD2
        elif param =='3': #학습
            cmd = EXTERNAL_CMD3
        print(f"실행.... 정보.............{cmd}")
        if( cmd == "") :
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "failed"
                log_fh.write(f"\n[ERR] No Cmd info \n")
            return
        elif(param == "1") : ## 동영상 추출
            cmd = cmd + ["--videos", file_path, "--every", "5"]
        elif(param =="2" ) : ## label extraction
            cmd = cmd + ["--target", target,]
        elif(param =="3" ) : ## 학습
            target =f"{job_id}"
            cmd = cmd + ["--job_id", job_id,]
            cmd = cmd + ["--data_info", data_info,]
        #elif(param == "2" ) :
        #    if
        #        cmd = cmd + ["--videos", file_path, "--every", "5"]
        print(f"실행명령....> {cmd}")
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh, cwd=BASE_DIR)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "running"
            JOBS[job_id]["pid"] = proc.pid
            JOBS[job_id]["process"] = proc

        print(f"waiting............................{proc}")
        rc = proc.wait()

        print(f"rc = proc.wait() ... ............................{rc}")
        with JOBS_LOCK:
            JOBS[job_id]["returncode"] = rc
            JOBS[job_id]["status"] = "completed" if rc == 0 else "failed"
        print("end of job............................")
        log_fh.write(f"\n[INFO] Process finished with return code {rc}\n")
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
        log_fh.write(f"\n[ERR] Exception: {e}\n")
        print(f"error : {e}")
    finally:
        log_fh.close()

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


# ====== Flask 앱 ======
app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def index():
    global CONFIG
    if MODEL == None :
        init_model()
    # 외부 파일(templates/index.html)을 읽어서 렌더링
    CONFIG = get_config()

    print(f"{CONFIG['data']['collect_root']}")
    return render_template("index.html")
@app.route("/data")
def data():
    # 외부 파일(templates/index.html)을 읽어서 렌더링
    print("index")
    return render_template("index_dataset.html")

@app.route("/img_data")
def img_data():
    # 외부 파일(templates/index.html)을 읽어서 렌더링
    print("index")
    return render_template("img_dataset.html")
@app.route("/train")
def train():
    # 외부 파일(templates/index.html)을 읽어서 렌더링
    print("index")
    return render_template("train.html")

@app.route("/yolo_test")
def yolo_test():
    # 외부 파일(templates/index.html)을 읽어서 렌더링
    print("index")
    return render_template("yolo_test.html")


@app.route("/setup")
def setup_env():
    # 외부 파일(templates/index.html)을 읽어서 렌더링
    print("index")
    return render_template("setup.html")

@app.route("/seoul")
def seoul():
    # 외부 파일(templates/index.html)을 읽어서 렌더링
    print("index")
    return render_template("seoul.html")

@app.route("/collection")
def collection():
    # 외부 파일(templates/index.html)을 읽어서 렌더링
    print("index")
    return render_template("collection.html")


CONFIG_PATH = get_config_path() #  "./training_config.json"  # 저장할 파일 경로



@app.route("/api/get_config", methods=["GET"])
def api_get_config():
    config = get_config()
    return jsonify(config)

@app.route("/api/config", methods=["POST"])
def receive_config():
    try:
        # JSON 파싱
        data = request.get_json(force=True)

        # 로그 출력 (서버 콘솔 확인용)
        print("=== Received Config ===")
        print(data)

        # 필요한 값 접근 예시
        collect_root = data.get("data", {}).get("collect_root")
        model_path = data.get("train", {}).get("model_path")
        edge_enable = data.get("edge", {}).get("enable")

        print(f"수집경로: {collect_root}")
        print(f"모델경로: {model_path}")
        print(f"엣지배포 활성화: {edge_enable}")

        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)

        # JSON 파일로 저장 (UTF-8, 보기 좋게 indent=2)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print("✅ Config 저장 완료:", CONFIG_PATH)

        return jsonify({
            "status": "ok",
            "message": f"Config saved to {CONFIG_PATH}"
        }), 200

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/upload_zip', methods=['POST'])
def upload_zip():
    try:
        if 'file' not in request.files:
            return jsonify({'status':'error', 'message':'No file uploaded'}), 400

        file = request.files['file']
        if not file.filename.endswith('.zip'):
            return jsonify({'status':'error', 'message':'Only ZIP files allowed'}), 400

        # 고유 파일명 생성 (예: upload_20251014_120355.zip)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"upload_{timestamp}.zip"
        save_path = os.path.join(UPLOAD_DIR, save_name)
        file.save(save_path)

        # 압축 해제 디렉토리 생성
        extract_path = os.path.join(EXTRACT_DIR, f"unzipped_{timestamp}")
        os.makedirs(extract_path, exist_ok=True)

        # ZIP 해제
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        # 해제된 파일 목록 반환
        extracted_files = []
        for root, dirs, files in os.walk(extract_path):
            for f in files:
                rel_path = os.path.relpath(os.path.join(root, f), extract_path)
                extracted_files.append(rel_path)

        #extract_label(extract_path)


        return jsonify({
            'status': 'ok',
            'message': f'{len(extracted_files)} files extracted',
            'upload_path': save_path,
            'extract_path': extract_path,
            'files': extracted_files
        })
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({'status': 'error', 'message': str(e)}), 500



@app.get("/api/train_dataset")
def train_dataset():

    model = request.args.get("model","")
    save_path = Path("train")
    param_job_id = request.args.get("job_id","")
    print(f"train dataset {param_job_id}")
    job_id = new_job(save_path)
    t = threading.Thread(target=run_job_async, args=(job_id, "3", param_job_id), daemon=True)
    t.start()
    print(f"..........threading......{job_id}")
    return jsonify({"job_id": job_id})

@app.get("/api/stop/<job_id>")
def stop_train(job_id):
    print(f"프로세스를 종료합니다. {job_id}")
    status = stop_job(job_id)
    print(f"{status}")
    return jsonify({"status": status})

@app.post("/api/upload")
def upload():
    f = request.files.get("video")
    if not f:
        return jsonify({"error": "no file"}), 400
    print("upload...................")
    filename = secure_filename(f.filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"not allowed extension: {ext}"}), 400

    save_path = UPLOAD_DIR +"/"+ filename
    f.save(save_path)
    print(f"upload...................{save_path}")
    job_id = new_job(save_path)
    t = threading.Thread(target=run_job_async, args=(job_id, "1",""), daemon=True)
    t.start()
    print(f"upload...................threading......{job_id}")

    conn = getCon()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT f.id, f.frame_index, f.image_path,
                            (SELECT COUNT(1) FROM boxes b WHERE b.frame_id=f.id) AS box_count
                    FROM frames f
                    WHERE f.job_id=%s
                    ORDER BY f.id desc
                    LIMIT %s OFFSET %s + %s
                    """,
                    ( job_id, 0, 0, limit, ))
    except psycopg2.Error as e:
        logging.error("PG error: %s", e)
    finally:
        conn.close()


    return jsonify({"job_id": job_id})

@app.get("/api/status/<job_id>")
def status(job_id):
    with JOBS_LOCK:
        meta = JOBS.get(job_id)
    if not meta:
        return jsonify({"error": "job not found"}), 404
    return jsonify({
        "status": meta["status"],
        "pid": meta["pid"],
        "returncode": meta["returncode"],
        "file": meta["file_path"],
        "log": meta["log_path"],
    })

@app.get("/api/log/<job_id>")
def log(job_id):
    tail = int(request.args.get("tail", "200"))
    with JOBS_LOCK:
        meta = JOBS.get(job_id)
    if not meta:
        return "job not found", 404
    path = Path(meta["log_path"])
    if not path.exists():
        return "(log pending)", 200
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    ret = "\n".join(lines[-tail:]), 200, {"Content-Type": "text/plain; charset=utf-8"}
    #print(f"{ret}")
    return ret

@app.get("/api/frames")
def list_frames():

    print("test..........")
    job_id = request.args.get("job_id","")

    print(f"job_id list_frames ----> {job_id}")
    if not job_id: return jsonify({"error":"job_id required"}), 400
    after = int(request.args.get("after","0")); limit = int(request.args.get("limit","10"))
    conn = getCon()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:

                cur.execute("""
                    SELECT f.id, f.frame_index, f.image_path,
                            (SELECT COUNT(1) FROM boxes b WHERE b.frame_id=f.id) AS box_count
                    FROM frames f
                    WHERE f.job_id=%s
                    ORDER BY f.id ASC
                    LIMIT %s OFFSET %s
                    """,
                    ( job_id, (after +1)* limit, after*limit, ))
                rows = cur.fetchall()

    except psycopg2.Error as e:
        logging.error("PG error: %s", e)
    finally:
        conn.close()

    for r in rows:
        r["image_url"]   = f"/api/frame/{r['id']}/image"
        r["overlay_url"] = f"/api/frame/{r['id']}/overlay"

    return jsonify(rows)

@app.get("/api/boxes/<int:frame_id>")
def list_boxes(frame_id:int):

    print(f"frame_id   {frame_id}")
    after = int(request.args.get("after","0")); limit = int(request.args.get("limit","100"))
    conn = getCon()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:

                cur.execute("""
                    SELECT id, frame_id, cls, conf, x1, y1, x2, y2
                    FROM boxes
                    WHERE frame_id=%s
                    LIMIT %s OFFSET %s
                    """,
                    ( frame_id, (after +1)* limit, after*limit, ))
                rows = cur.fetchall()

    except psycopg2.Error as e:
        logging.error("PG error: %s", e)
    finally:
        conn.close()

    return jsonify(rows)



@app.get("/api/frame/<int:frame_id>")
def frame_detail(frame_id:int):

    print(f"frame_id   {frame_id}")

    conn = getCon()
    rows = []
    rows_box = []
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT id, job_id, frame_index, image_path, width, height, ts FROM frames WHERE id=%s", (frame_id,))
                rows = cur.fetchone()
                if not rows: return jsonify({"error":"frame not found"}), 404
                boxes = cur.execute(
                    """
                    SELECT a.id, a.cls, a.conf, a.x1,a.y1,a.x2,a.y2, a.frame_id
                    FROM boxes a WHERE a.frame_id=%s ORDER BY a.id
                    """,
                    ( frame_id,)
                )
                rows_box = cur.fetchall()
                print(rows_box)
    except psycopg2.Error as e:
        logging.error("PG error: %s", e)
    finally:
        conn.close()
    #print(rows)
    return jsonify({
        "id": rows["id"], "job_id": rows["job_id"], "frame_index": rows["frame_index"],
        "width" : rows["width"],"height" : rows["height"], "overlay_url"  : rows["image_path"],
        "boxes" : rows_box
    })



@app.patch("/api/box/<int:box_id>")
def update_box_psycopg(box_id: int):
    data = request.get_json(silent=True) or {}
    cols, vals = [], []

    if "cls" in data and isinstance(data["cls"], str):
        cols.append("cls=%s"); vals.append(data["cls"].strip())

    for k in ("conf","x1","y1","x2","y2"):
        if k in data and data[k] is not None:
            try:
                v = float(data[k]) if k=="conf" else int(data[k])
            except Exception:
                return (f"bad {k}", 400)
            cols.append(f"{k}=%s"); vals.append(v)

    if not cols:
        return ("no fields", 400)

    vals.append(int(box_id))
    sql = f"UPDATE boxes SET {', '.join(cols)} WHERE id=%s"

    conn = getCon()

    with conn:  # 트랜잭션
        with conn.cursor() as cur:
            cur.execute(sql, tuple(vals))
            if cur.rowcount == 0:
                return ("box not found", 404)
    return jsonify({"ok": True, "id": box_id})

@app.patch("/api/box/insert")
def insert_box_psycopg():
    data = request.get_json(silent=True) or {}
    cols, vals = [], []
    print(f"{data}")


    sql = """
        INSERT INTO boxes(frame_id, cls, conf, x1, y1, x2, y2)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        returning id
        """

    conn = getCon()
    rows = []
    box_id=-1
    try :
        with conn:  # 트랜잭션
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                            INSERT INTO boxes(frame_id, cls, conf, x1, y1, x2, y2)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            returning id
                            """,
                            ( str(data['frame_id']), str(data["cls"]), str(data["conf"]),
                    float(data["x1"]), float(data["y1"]), float(data["x2"]), float(data["y2"]),)
                            )
                box_id = cur.fetchone()
    except psycopg2.Error as e:
            logging.error("PG error: %s", e)
    finally:
        conn.close()
    return jsonify({"ok": True, "id": box_id})

@app.get("/api/job/<job_id>/delete")
def delete_job(job_id):
    conn = getCon()

    print(f"delete job {job_id}")
    try:
        with conn:  # 트랜잭션
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                            DELETE FROM job
                            WHERE job_id=%s
                            """,
                            (str(job_id),)
                            )
                print(f"{job_id} is deleted")
                cur.execute("""
                            SELECT * FROM  frames
                            WHERE job_id=%s
                            """,
                            (str(job_id),)
                            )
                rows = cur.fetchall()
                for r in rows :
                    cur.execute("""
                            DELETE FROM boxes
                            WHERE frame_id=%s
                            """,
                            (str(r['id']),)
                            )
                    print(f"boxes in frame {r['id']} is deleted")
                cur.execute("""
                            DELETE  FROM  frames
                            WHERE job_id=%s
                            """,
                            (str(job_id),)
                            )
                print(f"frame in job {job_id} is deleted")
    except psycopg2.Error as e:
            logging.error("PG error: %s", e)
    finally:
        conn.close()
    return jsonify({"ok": True, "id": job_id})

@app.get("/api/frame/<frame_id>/delete")
def delete_frame(frame_id):
    conn = getCon()

    print(f"delete frame_id {frame_id}")
    try:
        with conn:  # 트랜잭션
            with conn.cursor(cursor_factory=RealDictCursor) as cur:

                cur.execute("""
                        DELETE FROM boxes
                        WHERE frame_id=%s
                        """,
                        (str(frame_id),)
                        )
                print(f"boxes in frame {frame_id} is deleted")
                cur.execute("""
                            DELETE  FROM  frames
                            WHERE id=%s
                            """,
                            (str(frame_id),)
                            )
                print(f"frame  {frame_id} is deleted")
    except psycopg2.Error as e:
            logging.error("PG error: %s", e)
    finally:
        conn.close()
    return jsonify({"ok": True, "id": frame_id})

@app.patch("/api/box/<int:box_id>/delete")
def delete_box(box_id: int):

    conn = getCon()
    print(f"box_id    {box_id}")
    try:
        with conn:  # 트랜잭션
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                            DELETE FROM boxes
                            WHERE id=%s
                            """,
                            (str(box_id),)
                            )
    except psycopg2.Error as e:
            logging.error("PG error: %s", e)
    finally:
        conn.close()
    return jsonify({"ok": True, "id": box_id})


@app.get("/api/frame/<int:frame_id>/image")
def frame_image(frame_id:int):
    with engine.connect() as conn:
        row = conn.execute(text("SELECT image_path FROM frames WHERE id=:id"), {"id": frame_id}).first()
    if not row: return "not found", 404
    p = Path(row[0]);
    if not p.exists(): return "missing file", 404
    return send_file(str(p), mimetype="image/jpeg")

@app.get("/api/frame/<int:frame_id>/overlay")
def frame_overlay(frame_id:int):
    with engine.connect() as conn:
        fr = conn.execute(text("SELECT image_path,width,height FROM frames WHERE id=:id"), {"id": frame_id}).first()
        if not fr: return "not found", 404
        boxes = conn.execute(text("SELECT cls, conf, x1,y1,x2,y2 FROM boxes WHERE frame_id=:id"), {"id": frame_id}).all()

    img = cv2.imread(fr[0])
    if img is None: return "missing image", 404
    H, W = img.shape[:2]
    for (cls, conf, x1,y1,x2,y2) in boxes:
        x1 = max(0, min(int(x1), W-1)); y1 = max(0, min(int(y1), H-1))
        x2 = max(0, min(int(x2), W-1)); y2 = max(0, min(int(y2), H-1))
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
        label = f"{cls} {conf:.2f}" if conf is not None else str(cls)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1-4)
        cv2.rectangle(img, (x1, y_text-th-base), (x1+tw, y_text+base), (0,0,0), -1)
        cv2.putText(img, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok: return "encode error", 500
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")

@app.get("/api/frames_all")
def list_frames_all():
    print("test..........")
    job_id = request.args.get("job_id","-1")
    limit = int(request.args.get("limit","10"))
    after = int(request.args.get("page","0"));
    conn = getCon()
    rows = []
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    WITH filtered AS (
                        SELECT
                            f.id,
                            f.frame_index,
                            f.image_path,
                            COUNT(*) OVER ()               AS total_count,
                            -- 내림차순(id DESC) 기준의 전체 순번(1부터 시작)
                            ROW_NUMBER() OVER (ORDER BY f.id DESC) AS pos_desc,
                            -- 각 프레임의 박스 개수
                            (SELECT COUNT(*) FROM boxes b WHERE b.frame_id = f.id) AS box_count
                        FROM frames f
                        WHERE f.job_id = %s
                        )
                        SELECT
                        id,
                        frame_index,
                        image_path,
                        box_count,
                        total_count,
                        pos_desc,                          -- 전체에서의 위치(내림차순 1..N)
                        (total_count - pos_desc + 1) AS pos_asc  -- 오름차순 1..N로 보고 싶으면 이 컬럼 사용
                        FROM filtered
                        ORDER BY id DESC
                        LIMIT %s OFFSET %s

                    """,
                    ( str(job_id), limit, after*limit, ))
                rows = cur.fetchall()

    except psycopg2.Error as e:
            logging.error("PG error: %s", e)
    finally:
        conn.close()


    for r in rows:
        r["image_url"]   = f"/api/frame/{r['id']}/image"
        r["overlay_url"] = f"/api/frame/{r['id']}/overlay"

    return jsonify(rows)


@app.get("/api/get_frame_cnt")
def job_frame_cnt():

    job_id = request.args.get("job_id","-1")
    #print(f"get_frame_cnt job_id --> {job_id}")
    conn = getCon()
    rows = []
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT COUNT(1)  AS total_count
                    FROM frames f
                    WHERE job_id = %s
                    """,
                    ( job_id, ))
                rows = cur.fetchone()

    except psycopg2.Error as e:
            logging.error("PG error: %s", e)
    finally:
        conn.close()
    print(rows)
    return jsonify(rows)

@app.get("/api/get_job_cnt")
def job_cnt():

    conn = getCon()
    rows = []
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT count(1)  AS total_count from
                    job
                    """,
                    (  ))
                rows = cur.fetchone()

    except psycopg2.Error as e:
            logging.error("PG error: %s", e)
    finally:
        conn.close()

    print(f"{rows}")

    return jsonify(rows)

@app.get("/api/get_job_list")
def job_list():

    limit = int(request.args.get("limit","10"))
    after = int(request.args.get("page","0"));
    conn = getCon()
    rows = []
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT job_id,file_name, to_char(ts,'YYYY-MM-DD') ts,ext_num,
                           (select COUNT(1)  AS total_count from frames f where f.job_id = j.job_id) total_count
                    FROM job j
                    order by ts desc
                    LIMIT %s  OFFSET %s
                    """,
                    (    limit, (after*limit), ))
                rows = cur.fetchall()

    except psycopg2.Error as e:
            logging.error("PG error: %s", e)
    finally:
        conn.close()

    return jsonify(rows)

@app.get("/api/re_extract/<job_id>")
def re_extract(job_id):

    print("re_extract job_id "+job_id)
    conn = getCon()
    rows = []
    boxes = []
    print(f"connecting...........{conn} ")
    label_path =""
    image_path =""
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * from frames
                    where job_id = %s
                    """,
                    (    job_id, ))
                rows = cur.fetchall()
            #print(f"for....................................{rows}")
            if(rows.count) :
                label_path = make_label_path(job_id, 'labels')
                image_path = make_label_path(job_id, 'images')

            for r in rows :

                with conn.cursor(cursor_factory=RealDictCursor) as cur2:
                    cur2.execute("""
                        SELECT * from boxes
                        where frame_id = %s
                        """,
                        ( r["id"],))
                    boxes = cur2.fetchall()
                    extract_data(job_id,r, boxes, label_path)
                    move_image(r, image_path)

            update_extract_num(job_id)
    except psycopg2.Error as e:
            logging.error("PG error: %s", e)
    finally:
        conn.close()

    return jsonify(rows)


def move_image(info, image_path) :
    path = info['image_path']
    shutil.copy(path, image_path)

import os
import shutil

def make_label_path(job_id, base_name="labels"):
    base_path = f'./static/yolo_dataset/{job_id}/train'
    origin_dir = os.path.join(base_path, base_name)

    # 기존 디렉토리가 존재할 경우 이름을 변경한다.
    if os.path.exists(origin_dir):
        counter = 1
        new_dir = os.path.join(base_path, f"{base_name}_{counter}")

        # 중복 이름 피하기
        while os.path.exists(new_dir):
            counter += 1
            new_dir = os.path.join(base_path, f"{base_name}_{counter}")

        # 기존 디렉토리 이름 변경
        shutil.move(origin_dir, new_dir)
        print(f"[INFO] 기존 디렉토리 → {new_dir} 로 이름 변경됨")

    # 기본 디렉토리를 새로 만든다.
    os.makedirs(origin_dir, exist_ok=True)
    print(f"[INFO] 새 기본 디렉토리 생성: {origin_dir}")

    return origin_dir


def extract_data(job_id, frame, boxes, label_path) :
    print(f"frameinfo {frame} label_path {label_path} boxes {boxes}")
    path = frame['image_path']

    size = (frame['width'], frame['height'])
    file_name = os.path.splitext(os.path.basename(path))[0]
    dir_name = os.path.dirname(label_path)
    print(f"path ....... {path} {file_name}")
    line=[]
    print("변환전.....")

    for box  in boxes :
        ibox = (box['cls'], box['x1'], box['y1'], box['x2'], box['y2'])
        l = convert_to_yolo(size, ibox)
        #l= str(box['cls']) +' ' +str( box['x1'])+' ' + str(box['y1'])+' '+str(box['x2'])+' '+str(box['y2'])+' '
        line.append(l)
        print(l)
    print("변환완료.....")
    file_name = label_path + "/" + file_name + ".txt"
    print(f"{file_name}")
    with open(file_name, "w") as f:
        for l in line:
            f.write(l + "\n")

    return True

def update_extract_num(job_id):
    print(f"extract_num   {job_id}")
    sql = f"UPDATE job SET ext_num=ext_num+1 WHERE job_id=%s"
    conn = getCon()
    try :
        with conn:  # 트랜잭션
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                print(f"실행하기 준비{sql}")
                cur.execute(
                            sql,
                           (job_id,)
                            )
            if cur.rowcount == 0:
                return ("box not found", 404)
    except psycopg2.Error as e:
        print(f"에러 발생.....  {e}")
        logging.error("PG error: %s", e)

def convert_to_yolo(size, box):
    """
    size: (width, height)
    box: (class_id, x_min, y_min, x_max, y_max)
    """
    #print("-----------------------convert_to_yolo")
    image_w, image_h = size
    class_id, x_min, y_min, x_max, y_max = box

    # 중심 좌표
    x_center = float(x_min + x_max) / 2.0 / float(image_w)
    y_center = float(y_min + y_max) / 2.0 / float(image_h)
    # 박스 크기
    w = float(x_max - x_min) / float(image_w)
    h = float(y_max - y_min) / float(image_h)

    return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"



@app.get("/api/model/list")
def model_list():
    limit = int(request.args.get("limit","10"))
    after = int(request.args.get("page","0"))
    conn = getCon()
    rows = []
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT job_id,file_name, to_char(ts,'YYYY-MM-DD') ts,ext_num,
                           (select COUNT(1)  AS total_count from frames f where f.job_id = j.job_id) total_count
                    FROM job j
                    order by ts desc
                    LIMIT %s  OFFSET %s
                    """,
                    (    limit, (after*limit), ))
                rows = cur.fetchall()

    except psycopg2.Error as e:
            logging.error("PG error: %s", e)
    finally:
        conn.close()

    return jsonify(rows)


@app.get("/sys_ok")
def healthz():
    return "ok", 200


def init_model() :
    # 1. 모델 로드
    global MODEL
    global config
    print(f"model loading................{config['train']['test_model']}")

    model_path = 'yolov8n.pt'
    MODEL = YOLO(model_path)
    
    print("Loaded model:", model_path)

# static 디렉토리 생성
if not os.path.exists("static"):
    os.makedirs("static")

# 2. 예측 API
@app.route("/predict_api/", methods=["POST"])
def predict_api():
    if MODEL == None :
       init_model()

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    #print(f"{MODEL}")
    results = MODEL.predict(img)

    # 예측 결과 시각화
    result_img = results[0].plot()
    result_pil = Image.fromarray(result_img)
    img_name = f"{uuid.uuid4().hex}.jpg"
    img_path = os.path.join("static", img_name)
    result_pil.save(img_path)

    # 예측 박스 정보
    predictions = []
    for result in results:
        for box in result.boxes:
            predictions.append({
                "class": MODEL.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })

    return jsonify({
        "image_url": f"/static/{img_name}",
        "predictions": predictions
    })

# 3. 정적 파일 서빙
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5510, debug=True)
