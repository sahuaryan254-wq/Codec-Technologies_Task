import cv2
from flask import Flask, render_template_string, Response, jsonify
from face_detection import FaceDetector
from emotion_recognition import EmotionModel
from depression_analysis import DepressionAnalyzer
from database import DatabaseManager

app = Flask(__name__)

face_detector = FaceDetector()
emotion_model = EmotionModel()
depression_analyzer = DepressionAnalyzer()
database = DatabaseManager()

HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Depression Detection Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; background: #f2f3f7; color: #333; padding: 24px; }
    .card { background: white; border-radius: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.08); padding: 20px; max-width: 640px; margin: auto; }
    .status { font-size: 1.1rem; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Depression Detection Dashboard</h1>
    <p class="status">Live video feed and emotion analysis will appear here.</p>
  </div>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def status():
    return jsonify({'status': 'ready'})

@app.route('/api/records/<user_id>')
def records(user_id):
    rows = database.fetch_user_records(user_id)
    return jsonify([{
        'id': row[0],
        'user_id': row[1],
        'emotion': row[2],
        'score': row[3],
        'risk_score': row[4],
        'recorded_at': row[5],
    } for row in rows])

def gen_video_stream():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError('Unable to access camera')

    while True:
        success, frame = capture.read()
        if not success:
            break

        faces = face_detector.detect_faces(frame)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            try:
                emotion = emotion_model.predict_emotion(face)
            except Exception:
                emotion = 'Unknown'
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
