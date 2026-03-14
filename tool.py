import cv2
import mediapipe as mp
import numpy as np
import math
import time
import sys

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("Error: Could not import mediapipe.tasks.")
    sys.exit(1)

class AICircleTool:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO)
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.num_sides = 3
        self.center = (960, 360) # Spostato sulla destra
        self.radius = 150
        self.max_sides = 20
        self.min_sides = 3
        
        self.prev_distance = 0
        self.distance_threshold = 0.02
        self.gesture_cooldown = 0
        self.cooldown_frames = 10
        self.last_gesture = ""
        self.gesture_timer = 0
        
        self.HAND_CONNECTIONS = frozenset([
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ])

    def calculate_distance(self, point1, point2):
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def draw_transparent_rect(self, image, top_left, bottom_right, color, alpha):
        overlay = image.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def draw_polygon(self, image, num_sides, center, radius):
        if num_sides < 3:
            return
            
        points = []
        for i in range(num_sides):
            angle = 2 * math.pi * i / num_sides - math.pi / 2
            x = int(center[0] + radius * math.cos(angle))
            y = int(center[1] + radius * math.sin(angle))
            points.append((x, y))

        points_array = np.array(points, np.int32)
        
        overlay = image.copy()
        cv2.fillPoly(overlay, [points_array], (255, 191, 0)) 
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        cv2.polylines(image, [points_array], True, (255, 255, 255), 2, cv2.LINE_AA)
        
        for point in points:
            cv2.circle(image, point, 6, (0, 255, 255), -1, cv2.LINE_AA)

    def draw_ui(self, image):
        h, w, _ = image.shape
        
        self.draw_transparent_rect(image, (20, 20), (350, 130), (30, 30, 30), 0.7)
        
        cv2.line(image, (20, 20), (20, 130), (0, 255, 255), 4)
        
        cv2.putText(image, f"SIDES: {self.num_sides}", (40, 60), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        polygon_names = {
            3: "Triangle", 4: "Square", 5: "Pentagon", 6: "Hexagon",
            7: "Heptagon", 8: "Octagon", 9: "Enneagon", 10: "Decagon"
        }
        name = polygon_names.get(self.num_sides, f"Polygon ({self.num_sides})")
        cv2.putText(image, name.upper(), (40, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        self.draw_transparent_rect(image, (20, h - 170), (400, h - 20), (30, 30, 30), 0.7)
        
        cv2.putText(image, "CONTROLS", (40, h - 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "Pinch to DECREASE sides", (40, h - 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "Spread to INCREASE sides", (40, h - 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "Press 'Q' to EXIT", (40, h - 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        if self.gesture_timer > 0:
            self.gesture_timer -= 1
            color = (0, 255, 0) if "Spread" in self.last_gesture else (0, 0, 255)
            cv2.putText(image, self.last_gesture, (w - 350, 80), 
                       cv2.FONT_HERSHEY_DUPLEX, 1, color, 2, cv2.LINE_AA)

    def process_gesture(self, landmarks):
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
            return
        
        thumb_tip = landmarks[4] 
        index_tip = landmarks[8]
        
        current_distance = self.calculate_distance(thumb_tip, index_tip)
        
        if self.prev_distance > 0:
            distance_change = current_distance - self.prev_distance
            
            if distance_change > self.distance_threshold:
                if self.num_sides < self.max_sides:
                    self.num_sides += 1
                    self.gesture_cooldown = self.cooldown_frames
                    self.last_gesture = "EXPANDING (+)"
                    self.gesture_timer = 20
            
            elif distance_change < -self.distance_threshold:
                if self.num_sides > self.min_sides:
                    self.num_sides -= 1
                    self.gesture_cooldown = self.cooldown_frames
                    self.last_gesture = "SHRINKING (-)"
                    self.gesture_timer = 20
        
        self.prev_distance = current_distance
    
    def draw_landmarks(self, image, landmarks):
        h, w, _ = image.shape
        pixel_landmarks = []
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            pixel_landmarks.append((cx, cy))
            
        for connection in self.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            cv2.line(image, pixel_landmarks[start_idx], pixel_landmarks[end_idx], (100, 100, 100), 1, cv2.LINE_AA)
            
        for cx, cy in pixel_landmarks:
            cv2.circle(image, (cx, cy), 3, (200, 200, 200), -1, cv2.LINE_AA)

    def run(self):
        print("PINCH POLYGON STARTED")
        print("Press 'q' to exit.")
        
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - start_time) * 1000)
            detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            
            self.draw_polygon(frame, self.num_sides, self.center, self.radius)
            
            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    self.draw_landmarks(frame, hand_landmarks)
                    self.process_gesture(hand_landmarks)
                    
                    h, w, _ = frame.shape
                    thumb_tip = hand_landmarks[4]
                    index_tip = hand_landmarks[8]
                    thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                    index_pos = (int(index_tip.x * w), int(index_tip.y * h))
                    
                    dist_px = math.hypot(thumb_pos[0]-index_pos[0], thumb_pos[1]-index_pos[1])
                    line_color = (0, 255, 0) if dist_px > 100 else (0, 0, 255)
                    
                    cv2.line(frame, thumb_pos, index_pos, line_color, 2, cv2.LINE_AA)
                    cv2.circle(frame, thumb_pos, 8, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame, index_pos, 8, (255, 255, 255), 2, cv2.LINE_AA)
            
            self.draw_ui(frame)
            cv2.imshow('PINCH POLYGON', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        app = AICircleTool()
        app.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
