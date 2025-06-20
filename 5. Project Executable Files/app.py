from flask import Flask, request, jsonify, render_template
import os
import logging
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the pre-trained xception_model model
try:
    model = load_model('5. Project Executable Files/xception_model.h5')
  # Make sure this file exists in your root directory
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error("Error loading model", exc_info=True)
    model = None

# Dictionary mapping class indices to card names
card_names = {
    0: 'ace of clubs', 1: 'ace of diamonds', 2: 'ace of hearts', 3: 'ace of spades',
    4: 'eight of clubs', 5: 'eight of diamonds', 6: 'eight of hearts', 7: 'eight of spades',
    8: 'five of clubs', 9: 'five of diamonds', 10: 'five of hearts', 11: 'five of spades',
    12: 'four of clubs', 13: 'four of diamonds', 14: 'four of hearts', 15: 'four of spades',
    16: 'jack of clubs', 17: 'jack of diamonds', 18: 'jack of hearts', 19: 'jack of spades',
    20: 'joker', 21: 'king of clubs', 22: 'king of diamonds', 23: 'king of hearts',
    24: 'king of spades', 25: 'nine of clubs', 26: 'nine of diamonds', 27: 'nine of hearts',
    28: 'nine of spades', 29: 'queen of clubs', 30: 'queen of diamonds', 31: 'queen of hearts',
    32: 'queen of spades', 33: 'seven of clubs', 34: 'seven of diamonds', 35: 'seven of hearts',
    36: 'seven of spades', 37: 'six of clubs', 38: 'six of diamonds', 39: 'six of hearts',
    40: 'six of spades', 41: 'ten of clubs', 42: 'ten of diamonds', 43: 'ten of hearts',
    44: 'ten of spades', 45: 'three of clubs', 46: 'three of diamonds', 47: 'three of hearts',
    48: 'three of spades', 49: 'two of clubs', 50: 'two of diamonds', 51: 'two of hearts',
    52: 'two of spades'
}

# Set image upload directory
target_img = os.path.join(os.getcwd(), '5. Project Executable Files', 'static', 'images')
# Route: Home page
@app.route('/')
def main_index():
    return render_template('index.html')

# Route: Input form page
@app.route('/input')
def input_page():
    return render_template('input.html')

# Route: Prediction logic
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            logging.error("No file part in the request")
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            logging.error("No file selected")
            return jsonify({'error': 'No selected file'}), 400

        if file:
            file_path = os.path.join(target_img, file.filename)
            logging.debug(f"Saving file to {file_path}")
            file.save(file_path)
            logging.debug(f"File saved to {file_path}")

            # Preprocess the image
            image = load_img(file_path, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = image / 255.0
            logging.debug("Image preprocessed")

            # Make prediction
            predictions = model.predict(image)
            logging.debug(f"Model prediction: {predictions}")
            predicted_class = np.argmax(predictions, axis=1)[0]
            card_name = card_names.get(predicted_class, "Unknown")
            logging.debug(f"Predicted class index: {predicted_class}, card name: {card_name}")

            return render_template('output.html', card=card_name, image_filename=file.filename)


    except Exception as e:
        logging.error("Error during prediction", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Main entry
if __name__ == '__main__':
    if not os.path.exists(target_img):
        os.makedirs(target_img)
    app.run(debug=True)
