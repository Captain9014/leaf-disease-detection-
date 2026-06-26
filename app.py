from flask import Flask, render_template, request, redirect, url_for, flash
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# =======================
# Flask App & Config
# =======================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

db = SQLAlchemy(app)
login_manager = LoginManager(app)


login_manager.login_view = "login"


# =======================
# Database Models
# =======================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    predictions = db.relationship('Prediction', backref='user', lazy=True)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    disease = db.Column(db.String(200))
    confidence = db.Column(db.Float)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))


# =======================
# Flask Login Loader
# =======================
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# =======================
# Load ML Model
# =======================
model = tf.keras.models.load_model("model/plant_model.h5")

dataset_path = "dataset"
class_names = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])
print("Loaded Classes:", class_names)


# =======================
# Disease Information
# =======================
disease_info = {
    "early blight": {"description":"Fungal disease affecting older leaves first.","symptoms":"Brown circular spots with yellow rings.","treatment":"Apply fungicide and remove infected leaves.","prevention":"Maintain proper spacing and avoid overwatering."},
    "late blight": {"description":"Serious fungal infection affecting leaves and stems.","symptoms":"Dark brown spots with white fungal growth.","treatment":"Use recommended fungicides immediately.","prevention":"Use resistant varieties and ensure good drainage."},
    "leaf mold": {"description":"Common disease in humid conditions.","symptoms":"Yellow spots on upper leaf surface.","treatment":"Apply copper-based fungicide.","prevention":"Improve air circulation."},
    "healthy": {"description":"The plant is healthy and disease-free.","symptoms":"Green leaves with no visible damage.","treatment":"No treatment needed.","prevention":"Maintain regular watering and nutrition."},
    "bacterial spot": {"description":"Bacterial infection causing dark water-soaked spots.","symptoms":"Small brown spots with yellow halo.","treatment":"Use copper-based bactericide spray.","prevention":"Avoid overhead watering and use disease-free seeds."}
}


# =======================
# Routes
# =======================

# Default Page → Login
@app.route('/')
def home():
    return redirect(url_for('login'))


# -----------------------
# Registration
# -----------------------
@app.route('/register', methods=['GET','POST'])
def register():

    if request.method == 'POST':

        username = request.form['username']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash("Username already exists!", "danger")
            return redirect(url_for('register'))

        new_user = User(
            username=username,
            password=generate_password_hash(password)
        )

        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please login.", "success")

        return redirect(url_for('login'))

    return render_template('register.html')


# -----------------------
# Login
# -----------------------
@app.route('/login', methods=['GET','POST'])
def login():

    if request.method == 'POST':

        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):

            login_user(user)

            flash("Login successful!", "success")

            return redirect(url_for('dashboard'))

        else:

            flash("Invalid credentials!", "danger")

    return render_template('login.html')


# -----------------------
# Logout
# -----------------------
@app.route('/logout')
@login_required
def logout():

    logout_user()

    flash("Logged out successfully.", "success")

    return redirect(url_for('login'))


# -----------------------
# Dashboard (Main Page)
# -----------------------
@app.route('/index')
@login_required
def dashboard():

    predictions = Prediction.query.filter_by(
        user_id=current_user.id
    ).order_by(Prediction.date.desc()).all()

    return render_template('index.html', predictions=predictions)


# -----------------------
# Prediction
# -----------------------
@app.route('/predict', methods=['POST'])
@login_required
def predict():

    if 'file' not in request.files:
        flash("No file uploaded", "danger")
        return redirect(url_for('dashboard'))

    file = request.files['file']

    if file.filename == '':
        flash("No file selected", "danger")
        return redirect(url_for('dashboard'))

    img = Image.open(file).convert("RGB")
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    score = prediction[0]

    predicted_index = np.argmax(score)

    confidence = round(float(np.max(score))*100,2)

    predicted_class = class_names[predicted_index].replace("__","_")

    if "_" in predicted_class:
        crop, condition = predicted_class.split("_",1)
    else:
        crop = predicted_class
        condition = "Unknown"

    crop = crop.replace("_"," ")
    condition = condition.replace("_"," ").strip()

    info = disease_info.get(condition.lower(), None)

    new_pred = Prediction(
        disease=condition.title(),
        confidence=confidence,
        user_id=current_user.id
    )

    db.session.add(new_pred)
    db.session.commit()

    return render_template(
        'index.html',
        predictions=[new_pred],
        crop=crop,
        condition=condition.title(),
        confidence=confidence,
        info=info
    )


# =======================
# Run App
# =======================
if __name__ == '__main__':

    with app.app_context():
        db.create_all()

    app.run(debug=True)