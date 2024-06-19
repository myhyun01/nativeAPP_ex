import sys
import cv2
import torch
import time
from PySide6.QtWidgets import QMainWindow, QApplication, QLabel
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer
import serial
from google.cloud import speech
import pyaudio
import wave
import os

from window_yolo_ui import Ui_MainWindow

cls_names = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
    'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Initialize model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        # Initialize serial communication
        self.ser = serial.Serial('COM3', 9600, timeout=1)

        # Initialize Google Cloud Speech-to-Text client
        self.client = speech.SpeechClient()

        # Initialize UI elements for displaying text and audio output
        self.text_output_label = QLabel(self)
        self.text_output_label.setGeometry(10, 10, 780, 30)
        self.text_output_label.setStyleSheet("QLabel { background-color : white; color : black; }")
        
        self.audio_output_label = QLabel(self)
        self.audio_output_label.setGeometry(10, 50, 780, 30)
        self.audio_output_label.setStyleSheet("QLabel { background-color : white; color : black; }")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(image)
            self.detect_objects(results, image)
            qimg = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qimg))

    def detect_objects(self, results, image):
        for pred in results.pred[0]:
            conf, x1, y1, x2, y2, class_idx = pred[4], int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3]), int(pred[5])
            if conf > 0.5:
                class_name = cls_names[class_idx]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                self.display_text(f"Detected {class_name} with confidence {conf:.2f}")

                if class_name == 'person':
                    self.robotAction(17)
                    self.display_text("안녕하신가 휴먼. 무엇을 도와줄까?", speak=True)
                    time.sleep(7)
                    self.robotAction(1)
                    time.sleep(1)
                elif class_name == 'bottle':
                    self.robotAction(30)
                    self.display_text("그 병안에는 어떤 맛있는 음료가 들어 있냐?", speak=True)
                    time.sleep(7)
                    self.robotAction(1)
                    time.sleep(1)

    def robotAction(self, no):
        print(f"Robot Action: {no}")
        if self.ser.is_open:
            exeCmd = bytearray([0xff, 0xff, 0x4c, 0x53, 0x00,
                                0x00, 0x00, 0x00, 0x30, 0x0c, 0x03,
                                no, 0x00, 100, 0x00])

            exeCmd[14] = sum(exeCmd[6:14]) & 0xFF
            self.ser.write(exeCmd)
            time.sleep(0.05)

    def closeEvent(self, event):
        if self.ser.is_open:
            self.ser.close()
        event.accept()

    def record_audio(self):
        chunk = 1024
        sample_format = pyaudio.paInt16
        channels = 1
        fs = 44100
        seconds = 5
        filename = "output.wav"

        p = pyaudio.PyAudio()

        print("Recording...")

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        frames = []

        for _ in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        print("Recording finished.")

        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

        return filename

    def speak(self, text):
        self.timer.stop()
        audio_file = self.record_audio()
        self.recognize_speech(audio_file)
        os.remove(audio_file)
        self.timer.start()

    def recognize_speech(self, audio_file):
        with open(audio_file, "rb") as audio:
            content = audio.read()
        
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="ko-KR",
        )

        response = self.client.recognize(config=config, audio=audio)

        for result in response.results:
            transcript = result.alternatives[0].transcript
            self.display_text(f"Transcript: {transcript}", display_audio=True)

    def display_text(self, text, display_audio=False, speak=False):
        if display_audio:
            self.audio_output_label.setText(text)
        else:
            self.text_output_label.setText(text)
        
        if speak:
            self.speak(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
