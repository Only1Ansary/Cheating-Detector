import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import glob


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


if not os.path.exists('captured_images'):
    os.makedirs('captured_images')


for file in glob.glob('captured_images/*.jpg'):
    os.remove(file)

class GazeDetectorApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Precision Gaze Detector")
        self.window.geometry("1200x800")
        self.window.configure(bg='white')
        
        self.cap = cv2.VideoCapture(0)
        _, frame = self.cap.read()
        self.img_h, self.img_w = frame.shape[:2]
        
        self.video_frame = tk.Label(self.window, bg='white', borderwidth=2, relief="groove")
        self.video_frame.place(x=20, y=20, width=self.img_w, height=self.img_h)
        
        
        self.status_text = tk.StringVar(value="Status: FOCUSED")
        self.status_color = "green"
        self.status_label = tk.Label(self.window, textvariable=self.status_text, font=('Helvetica', 14), bg='white', fg=self.status_color)
        self.status_label.place(x=self.img_w + 40, y=30)
        
        self.processing_frame = tk.Frame(self.window, bg='white')
        self.processing_frame.place(x=20, y=self.img_h + 40, width=self.img_w, height=100)
        
        self.create_processing_buttons()
        
        self.finish_btn = ttk.Button(self.window, text="Show Original Captures", command=lambda: self.show_captures())
        self.finish_btn.place(x=self.img_w//2 - 100, y=self.img_h + 160, width=200, height=40)
        
        self.focal_length = self.img_w
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.img_w/2],
            [0, self.focal_length, self.img_h/2],
            [0, 0, 1]
        ], dtype="double")
        
        self.dist_coeffs = np.zeros((4, 1))
        self.model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])
        
        self.MAX_HEAD_YAW = 63
        self.MAX_HEAD_PITCH = 165
        self.GAZE_CONSEC_FRAMES = 1
        self.looking_away_counter = 0
        self.looking_away = False
        self.captured_images = []
        
        self.update()
    
    def create_processing_buttons(self):
        self.sharp_btn = ttk.Button(self.processing_frame, text="Sharpened Images", 
                                  command=lambda: self.show_captures(processing='Sharp'))
        self.sharp_btn.grid(row=0, column=0, padx=10, pady=5, ipadx=10, ipady=5)
        
        self.contrast_btn = ttk.Button(self.processing_frame, text="Contrast Stretched", 
                                     command=lambda: self.show_captures(processing='Contrast'))
        self.contrast_btn.grid(row=0, column=1, padx=10, pady=5, ipadx=10, ipady=5)
        
        self.smooth_btn = ttk.Button(self.processing_frame, text="Smoothed Images", 
                              command=lambda: self.show_captures(processing='Smooth'))
        self.smooth_btn.grid(row=0, column=2, padx=10, pady=5, ipadx=10, ipady=5)

        
        self.hist_btn = ttk.Button(self.processing_frame, text="Histogram Equalized", 
                                  command=lambda: self.show_captures(processing='Histogram'))
        self.hist_btn.grid(row=0, column=3, padx=10, pady=5, ipadx=10, ipady=5)
    
    def update(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    image_points = np.array([
                        [face_landmarks.landmark[4].x * self.img_w, face_landmarks.landmark[4].y * self.img_h],
                        [face_landmarks.landmark[152].x * self.img_w, face_landmarks.landmark[152].y * self.img_h],
                        [face_landmarks.landmark[133].x * self.img_w, face_landmarks.landmark[133].y * self.img_h],
                        [face_landmarks.landmark[362].x * self.img_w, face_landmarks.landmark[362].y * self.img_h],
                        [face_landmarks.landmark[61].x * self.img_w, face_landmarks.landmark[61].y * self.img_h],
                        [face_landmarks.landmark[291].x * self.img_w, face_landmarks.landmark[291].y * self.img_h]
                    ], dtype="double")

                    _, rotation_vec, _ = cv2.solvePnP(
                        self.model_points,
                        image_points,
                        self.camera_matrix,
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    rmat, _ = cv2.Rodrigues(rotation_vec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                    pitch, yaw, roll = angles

                    head_moved = (abs(yaw) > self.MAX_HEAD_YAW) or (abs(pitch) < self.MAX_HEAD_PITCH)
                    
                    prev_state = self.looking_away
                    if head_moved:
                        self.looking_away_counter += 1
                        if self.looking_away_counter >= self.GAZE_CONSEC_FRAMES and not self.looking_away:
                            self.looking_away = True
                    else:
                        self.looking_away_counter = max(0, self.looking_away_counter - 1)
                        if self.looking_away_counter == 0 and self.looking_away:
                            self.looking_away = False


                    if self.looking_away:
                        self.status_text.set("Status: LOOKING AWAY")
                        self.status_label.config(fg="red")
                    else:
                        self.status_text.set("Status: FOCUSED")
                        self.status_label.config(fg="green")
                    
                    if self.looking_away and not prev_state:
                        self.capture_image(frame)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)
        
        self.window.after(10, self.update)
    
    def capture_image(self, frame):
        filename = f"captured_images/capture_{len(self.captured_images)+1}.jpg"
        cv2.imwrite(filename, frame)
        self.captured_images.append(filename)
    
    def apply_Sharpening(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]], dtype=np.float32)
    
        sharpened = cv2.filter2D(img, -1, kernel)
        return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))

    def apply_ContrastStretching(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        min_val = np.min(l)
        max_val = np.max(l)
        
        if max_val == min_val:
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        stretched = ((l - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        enhanced_lab = cv2.merge([stretched, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)

    def apply_Smoothing(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        kernel_size = 5
        pad = kernel_size // 2

        padded_img = np.pad(img_rgb, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)

        smoothed_img = np.zeros_like(img_rgb)
        for c in range(3):
            smoothed_img[:, :, c] = cv2.filter2D(padded_img[:, :, c], -1, kernel)[pad:-pad, pad:-pad]

        return Image.fromarray(smoothed_img)


    def apply_HistogramEqualization(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        hist = np.zeros(256, dtype=int)
        for value in l.flatten():
            hist[value] += 1

        cdf = np.zeros(256, dtype=int)
        cdf[0] = hist[0]
        for i in range(1, 256):
            cdf[i] = cdf[i - 1] + hist[i]

        cdf_min = np.min(cdf[np.nonzero(cdf)])
        total_pixels = l.size
        transformation = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            transformation[i] = np.round((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255).astype(np.uint8)

        l_equalized = transformation[l]

        equalized_lab = cv2.merge([l_equalized, a, b])
        equalized_img = cv2.cvtColor(equalized_lab, cv2.COLOR_LAB2RGB)

        return Image.fromarray(equalized_img)


    def show_captures(self, processing=None):
        if not self.captured_images:
            messagebox.showinfo("No Captures", "No images have been captured yet!")
            return
        
        top = tk.Toplevel(self.window)
        title = "Captured Images"
        if processing:
            title += f" - {processing.capitalize()}"
        top.title(title)
        
        canvas = tk.Canvas(top)
        scrollbar = ttk.Scrollbar(top, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        for idx, img_path in enumerate(self.captured_images):
            if processing == 'Sharp':
                img = self.apply_Sharpening(img_path)
            elif processing == 'Contrast':
                img = self.apply_ContrastStretching(img_path)
            elif processing == 'Smooth':
                img = self.apply_Smoothing(img_path)
            elif processing == 'Histogram':
                img = self.apply_HistogramEqualization(img_path)
            else:
                img = Image.open(img_path)
            
            if img is None:
                continue
                
            img.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(img)
            label = ttk.Label(scrollable_frame, image=photo)
            label.image = photo
            label.grid(row=idx//4, column=idx%4, padx=5, pady=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def on_closing(self):
        self.cap.release()
        self.window.destroy()


root = tk.Tk()
app = GazeDetectorApp(root)
root.protocol("WM_DELETE_WINDOW", app.on_closing)
root.mainloop()
