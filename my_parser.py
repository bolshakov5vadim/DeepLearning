from selenium.webdriver.common.keys import Keys#websocket
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import requests # для скачивания фото
import random, string # для названий файлов
import time # для ожидания подзагрузки
import os # для записи файлов


def download(url, path):
     
    letters = string.ascii_lowercase # имя файла
    filename = ''.join(random.choice(letters) for i in range(10))
    filename =  path+'/'+filename+'.png'
    try:
        response = requests.get(url)
    except Exception:
        return
    
    with open(filename, "wb") as f:
            f.write(response.content)
    
#chromedriver.exe ДОЛЖЕН СОВПАДАТЬ С ВЕРСИЕЙ БРАУЗЕРА

print("Pinterest_parser_v2.0.")
print("Print url.")
url=input()
urls=[]

print("Print parsing limit.")
limit=input()
limit=int(limit)


driver = webdriver.Chrome()
driver.get(url)# Настроили эмулятор
        
i=0 # Сломает цикл если конец страницы
old=0

while True: # цикл PAGE_DOWN
        
            html = driver.page_source # получить код страницы
            soup = BeautifulSoup(html, "html.parser") # распарсить

            img_tags = soup.findAll('img') # найти картинки
            
            for obj in img_tags: # Сохранить все URL
                    str_img=str(obj.get("src"))
                    if str_img.find('https:')==-1:str_img='https:'+str_img
                    if str_img not in urls: urls.append(str_img)


            html = driver.find_element(By.TAG_NAME, 'html')
            html.send_keys(Keys.PAGE_DOWN)
            # Нажать кнопку PAGE DOWN

            #Если focus-trap, удалить
            # try:
            #     trap_element = driver.find_element(By.CSS_SELECTOR, "trap_focus")
            #     if (trap_element != None): driver.execute_script("arguments.remove();", trap_element)       
            # except Exception as e: print(e)

            print(f"Links: {len(urls)}")

            # Ожидание загрузки
            time.sleep(0.5)

            # Нажатие кнопок
            #button = driver.find_element_by_class_name("button2 button2_size_l button2_theme_action button2_type_link button2_view_classic more__button i-bem button2_js_inited")
            #if (button.size() > 0 && button.get(0).isDisplayed()): button.click()
            
            print(f"Links: {len(urls)}")

            if (len(urls)>limit):
                print("len(urls) more, then requested")
                break
            
            i += 1 # Проверка каждые 8 нажатий
            if i % 8 == 0: 
                if (len(urls) == old):
                    print("len(urls) stabled")
                    break
                old = len(urls)  
    
    
# если путь не существует, создать
path = "folder"
if not os.path.isdir(path):
    os.makedirs(path)

for i, url in enumerate(urls):
    download(url, path)
    print(f"Downloaded {i}/{len(urls)} images")
