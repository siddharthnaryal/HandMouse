import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math

CLICK_DISTANCE = 50
DOUBLE_CLICK_TIMEOUT = 0.3
THUMBS_UP_TIME = 2.0
MOUSE_SMOOTHING = 0.2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

screen_w, screen_h = pyautogui.size()

last_click_time = 0
click_count = 0
last_mouse_x, last_mouse_y = screen_w//2, screen_h//2
thumbs_up_active = False
thumbs_up_start_time = 0

def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def is_thumbs_up(hand_landmarks, frame_width, frame_height):
    def get_y(landmark):
        return landmark.y * frame_height

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    thumb_up = get_y(thumb_tip) < get_y(thumb_ip) - 10

    index_folded = get_y(index_tip) > get_y(index_mcp) + 20
    middle_folded = get_y(middle_tip) > get_y(middle_mcp) + 20
    ring_folded = get_y(ring_tip) > get_y(ring_mcp) + 20
    pinky_folded = get_y(pinky_tip) > get_y(pinky_mcp) + 20

    return thumb_up and index_folded and middle_folded and ring_folded and pinky_folded


def main():
    global last_click_time, click_count, last_mouse_x, last_mouse_y
    global thumbs_up_active, thumbs_up_start_time
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    print(" Virtual Mouse Control ")
    print("Hand gestures:")
    print("- Move hand: Controls cursor position")
    print("- Single pinch: Left click")
    print("- Double pinch: Double click")
    print("- Thumbs-up for 2 seconds: Quit program")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        h, w = frame.shape[:2]
        current_time = time.time()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                thumb_pt = (int(thumb.x * w), int(thumb.y * h))
                index_pt = (int(index.x * w), int(index.y * h))
                distance = calculate_distance(thumb_pt, index_pt)

                if is_thumbs_up(hand_landmarks, w, h):
                    if not thumbs_up_active:
                        thumbs_up_active = True
                        thumbs_up_start_time = current_time
                    else:
                        if current_time - thumbs_up_start_time > THUMBS_UP_TIME:
                            quit_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                            print(f"Thumbs-up quit gesture detected at {quit_time}")
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                    
                    cv2.putText(frame, f"QUITTING: {int(THUMBS_UP_TIME - (current_time - thumbs_up_start_time))}s", 
                               (w//2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    thumbs_up_active = False
                
                if not thumbs_up_active:
                    target_x = np.interp(thumb.x, [0.1, 0.9], [0, screen_w])
                    target_y = np.interp(thumb.y, [0.1, 0.9], [0, screen_h])
                    
                    last_mouse_x = last_mouse_x * (1 - MOUSE_SMOOTHING) + target_x * MOUSE_SMOOTHING
                    last_mouse_y = last_mouse_y * (1 - MOUSE_SMOOTHING) + target_y * MOUSE_SMOOTHING
                    pyautogui.moveTo(last_mouse_x, last_mouse_y)
                    
                    if distance < CLICK_DISTANCE:
                        time_since_last_click = current_time - last_click_time
                        
                        if time_since_last_click < DOUBLE_CLICK_TIMEOUT:
                            click_count += 1
                        else:
                            click_count = 1
                        
                        last_click_time = current_time
                        
                        feedback_color = (0, 255, 255)
                        feedback_text = "CLICK"
                        
                        if click_count == 2:
                            pyautogui.doubleClick()
                            feedback_color = (0, 255, 0)
                            feedback_text = "DOUBLE CLICK"
                            click_count = 0
                        elif time_since_last_click > 0.3:
                            pyautogui.click()
                        
                        cv2.circle(frame, thumb_pt, 30, feedback_color, -1)
                        cv2.putText(frame, feedback_text, (thumb_pt[0]-70, thumb_pt[1]-40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2)
                
                    line_color = (0, 165, 255)
                    cv2.line(frame, thumb_pt, index_pt, line_color, 3)
                    for pt in [thumb_pt, index_pt]:
                        cv2.circle(frame, pt, 15, line_color, -1)
                        cv2.circle(frame, pt, 20, line_color, 2)

        if not thumbs_up_active:
            cv2.putText(frame, "Move Hand to Control Mouse", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Pinch to Click | Double-Pinch to Double-Click", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Thumbs-Up for 2s to Quit", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Hand Mouse", frame)
        if cv2.waitKey(5) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
