from flask import Flask, render_template, request, send_from_directory, url_for
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from werkzeug.utils import secure_filename
import os
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your trained model
model = tf.keras.models.load_model(r"C:\Users\nagin\OneDrive\Desktop\galactic gazetter\flask\planet_classifier.h5")

# Labels and planet metadata
labels = ['Earth', 'Jupiter', 'Makemake', 'Mars', 'Mercury', 'Moon',
          'Neptune', 'Pluto', 'Saturn', 'Uranus', 'Venus', 'Others']

planet_data = {
    'Mercury': {
        'distance': '57.9 million km',
        'day': '1407.6 hours',
        'year': '88 days',
        'position': '1st',
        'diameter': '4,879 km',
        'origin': 'Named after the Roman messenger god Mercury',
        'composition': 'Rocky planet',
        'gravity': '3.7 m/s²',
        'atmosphere': 'Very thin; mostly oxygen, sodium, hydrogen',
        'orbital_speed': '47.36 km/s',
        'moons': '0'
    },
    'Venus': {
        'distance': '108.2 million km',
        'day': '5832.5 hours',
        'year': '225 days',
        'position': '2nd',
        'diameter': '12,104 km',
        'origin': 'Named after the Roman goddess of love and beauty',
        'composition': 'Rocky planet',
        'gravity': '8.87 m/s²',
        'atmosphere': '96% CO₂, sulfuric acid clouds',
        'orbital_speed': '35.02 km/s',
        'moons': '0'
    },
    'Earth': {
        'distance': '149.6 million km',
        'day': '24 hours',
        'year': '365.25 days',
        'position': '3rd',
        'diameter': '12,742 km',
        'origin': 'From Old English and Germanic roots meaning “ground”',
        'composition': 'Rocky planet',
        'gravity': '9.81 m/s²',
        'atmosphere': '78% nitrogen, 21% oxygen',
        'orbital_speed': '29.78 km/s',
        'moons': '1 (Moon)'
    },
    'Mars': {
        'distance': '227.9 million km',
        'day': '24.6 hours',
        'year': '687 days',
        'position': '4th',
        'diameter': '6,779 km',
        'origin': 'Named after the Roman god of war',
        'composition': 'Rocky planet',
        'gravity': '3.71 m/s²',
        'atmosphere': 'Mostly CO₂, with nitrogen and argon',
        'orbital_speed': '24.07 km/s',
        'moons': '2 (Phobos and Deimos)'
    },
    'Jupiter': {
        'distance': '778.5 million km',
        'day': '9.9 hours',
        'year': '4333 days',
        'position': '5th',
        'diameter': '139,820 km',
        'origin': 'Named after the king of the Roman gods',
        'composition': 'Gas giant (hydrogen and helium)',
        'gravity': '24.79 m/s²',
        'atmosphere': 'Hydrogen, helium, ammonia',
        'orbital_speed': '13.07 km/s',
        'moons': '95 known'
    },
    'Saturn': {
        'distance': '1.43 billion km',
        'day': '10.7 hours',
        'year': '10,759 days',
        'position': '6th',
        'diameter': '116,460 km',
        'origin': 'Named after the Roman god of agriculture',
        'composition': 'Gas giant',
        'gravity': '10.44 m/s²',
        'atmosphere': 'Hydrogen, helium, methane',
        'orbital_speed': '9.69 km/s',
        'moons': '145 known'
    },
    'Uranus': {
        'distance': '2.87 billion km',
        'day': '17.2 hours',
        'year': '30,687 days',
        'position': '7th',
        'diameter': '50,724 km',
        'origin': 'Named after the Greek god of the sky',
        'composition': 'Ice giant',
        'gravity': '8.69 m/s²',
        'atmosphere': 'Hydrogen, helium, methane',
        'orbital_speed': '6.81 km/s',
        'moons': '27 known'
    },
    'Neptune': {
        'distance': '4.5 billion km',
        'day': '16.1 hours',
        'year': '60,190 days',
        'position': '8th',
        'diameter': '49,244 km',
        'origin': 'Named after the Roman god of the sea',
        'composition': 'Ice giant',
        'gravity': '11.15 m/s²',
        'atmosphere': 'Hydrogen, helium, methane',
        'orbital_speed': '5.43 km/s',
        'moons': '14 known'
    },
    'Pluto': {
        'distance': '5.9 billion km',
        'day': '153.3 hours',
        'year': '90,560 days',
        'position': '9th (dwarf)',
        'diameter': '2,377 km',
        'origin': 'Named after the Roman god of the underworld',
        'composition': 'Ice-rock mix',
        'gravity': '0.62 m/s²',
        'atmosphere': 'Thin; nitrogen, methane',
        'orbital_speed': '4.74 km/s',
        'moons': '5 (Charon, etc.)'
    },
    'Makemake': {
        'distance': '6.85 billion km',
        'day': 'Unknown',
        'year': '112,897 days',
        'position': 'Beyond Pluto',
        'diameter': '1,434 km',
        'origin': 'Named after the creator god of the Rapa Nui mythology',
        'composition': 'Ice dwarf',
        'gravity': 'Unknown',
        'atmosphere': 'Possibly methane',
        'orbital_speed': '4.41 km/s',
        'moons': '1 known'
    },
    'Moon': {
        'distance': '384,400 km from Earth',
        'day': '655.7 hours',
        'year': '27.3 days (orbital)',
        'position': 'N/A',
        'diameter': '3,474 km',
        'origin': 'From Old English "mōna"',
        'composition': 'Rocky body',
        'gravity': '1.62 m/s²',
        'atmosphere': 'None (exosphere)',
        'orbital_speed': '1.02 km/s',
        'moons': '0'
    },
    'Others': {
        'distance': 'Unknown',
        'day': 'Unknown',
        'year': 'Unknown',
        'position': 'Unknown',
        'diameter': 'Unknown',
        'origin': 'Unknown',
        'composition': 'Unknown',
        'gravity': 'Unknown',
        'atmosphere': 'Unknown',
        'orbital_speed': 'Unknown',
        'moons': 'Unknown'
    }
}

@app.route('/')
def Home():
    return render_template("index.html")

@app.route('/index.html')
def index():
    return render_template("index.html")

@app.route('/find.html')
def find():
    return render_template("find.html")

@app.route('/about.html')
def about():
    return render_template("about.html")

@app.route('/contact.html')
def contact():
    return render_template("contact.html")

@app.route('/output.html', methods=['GET', 'POST'])
def output():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        f.save(filepath)
        uploaded_image_url = f'uploads/{filename}'

        # Image preprocessing
        img = load_img(filepath, target_size=(128, 128))
        x = img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        # Predict
        preds = model.predict(x)
        predicted_class_index = np.argmax(preds, axis=-1)
        predicted_class = labels[predicted_class_index[0]]
        info = planet_data.get(predicted_class, planet_data['Others'])

        return render_template("output.html", predicted_class=predicted_class, info=info, uploaded_image=uploaded_image_url)

    return render_template("output.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=7000)
