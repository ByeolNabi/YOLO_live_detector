from ultralytics import YOLO

import cv2
import numpy as np
import pyautogui

# 인풋 할 사진의 간격을 조절하고 싶다면 sleep을 사용한다.
from time import sleep

# 객체 검출에 사용할 yolo 모델의 위치
model = YOLO("C:/yolotest/yolov8x.pt")

# 실시간으로 검출하고싶은 스크린의 위치
top, left, width, height = 100, 0, 800, 600

while True:
    # Get the screenshot of the desired region
    img = pyautogui.screenshot(region=(left, top, width, height))
    frame = np.array(img)

    ##sleep(1)
    
    # Convert the screenshot to a BGR image (이렇게 해야 정상적인 색으로 표현이 된다)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Show the frame on the screen
    cv2.imshow("Screen", frame)
        
    # YOLO로 객체 검출하고 결과창 띄우기
    results = model.predict(source=frame, show=True)
    print(results) 
    
    # Check if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Close the window
cv2.destroyAllWindows()






