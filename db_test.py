# psycopg2_example.py
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from config_env import get_config, get_config_path

# 환경변수 예: postgresql://user:pass@host:5432/dbname
DATABASE_URL = os.getenv(
    "DATABASE_URL",
     "postgresql+psycopg2://postgres:1234@192.168.0.231:5432/public"
)

def main():
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
    try:
        with conn:
            with conn.cursor() as cur:
                # 테이블 생성
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS demo (
                        id SERIAL PRIMARY KEY,
                        name TEXT NOT NULL,
                        score INTEGER NOT NULL CHECK (score >= 0),
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                # INSERT (파라미터 바인딩으로 안전하게)
                cur.execute(
                    "INSERT INTO demo (name, score) VALUES (%s, %s) RETURNING id;",
                    ("alice", 95)
                )
                new_id = cur.fetchone()[0]
                print("inserted id:", new_id)

            # SELECT (dict 형태로 받고 싶으면 RealDictCursor)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT id, name, score, created_at FROM demo ORDER BY id DESC LIMIT 5;")
                for row in cur.fetchall():
                    print(row)

    finally:
        conn.close()

if __name__ == "__main__":
    main()
