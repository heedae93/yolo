import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# 검색어 및 저장 폴더 설정
#pip install Selenium  webdriver_manager
search_url = "https://www.google.com/search?q=업소용+음식물+쓰레기통&udm=2"
save_dir = "foodbin_images"
os.makedirs(save_dir, exist_ok=True)

# 크롬 옵션 설정
chrome_options = Options()
chrome_options.add_argument("--headless")  # 브라우저 창 숨김 실행
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

# 크롬 드라이버 실행
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
driver.get(search_url)

# 페이지 스크롤 (이미지 더 로드되도록)
for i in range(5):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(200)

# 이미지 태그 수집
images = driver.find_elements(By.TAG_NAME, "img")

print(f"총 {len(images)}개 이미지 발견됨.")

# 이미지 다운로드
for idx, img in enumerate(images):
    try:
        src = img.get_attribute("src")
        if src and src.startswith("http"):
            img_data = requests.get(src).content
            with open(os.path.join(save_dir, f"img_{idx}.jpg"), "wb") as f:
                f.write(img_data)
            print(f"다운로드 완료: img_{idx}.jpg")
    except Exception as e:
        print("에러:", e)

driver.quit()
