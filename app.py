from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model machine learning H5
model = tf.keras.models.load_model('ModelHerbPedia.h5')  # Ganti dengan path menuju model Anda

# Deskripsi dan manfaat tanaman berdasarkan label
plant_info = {
    "Sirih": {
        "description": "Sirih adalah tanaman merambat yang banyak ditemui di Asia Tenggara. Daunnya biasanya digunakan dalam berbagai tradisi dan upacara, serta memiliki beragam manfaat kesehatan seperti membantu dalam perawatan mulut dan membunuh kuman.",
        "benefits": ["Memperbaiki kesehatan mulut", "Membunuh kuman dan bakteri", "Meredakan batuk"]
    },
    "Seledri": {
        "description": "Seledri adalah tanaman dengan batang hijau dan daun berukuran kecil yang sering digunakan sebagai bahan makanan dan bumbu masakan. Seledri juga dikenal memiliki banyak manfaat kesehatan seperti membantu menurunkan tekanan darah dan meningkatkan sistem pencernaan.",
        "benefits": ["Menurunkan tekanan darah", "Menyegarkan napas", "Menyediakan antioksidan"]
    },
    "Pepaya": {
        "description": "Pepaya adalah buah tropis yang kaya akan vitamin C, serat, dan enzim pencernaan. Pepaya tidak hanya lezat, tetapi juga memiliki banyak manfaat kesehatan seperti meningkatkan sistem kekebalan tubuh, meningkatkan pencernaan, dan menyediakan antioksidan.",
        "benefits": ["Meningkatkan sistem kekebalan tubuh", "Meningkatkan pencernaan", "Menyediakan antioksidan"]
    },
    "Pandan": {
        "description": "Pandan adalah tanaman yang banyak digunakan dalam masakan Asia Tenggara untuk memberikan aroma dan rasa. Selain itu, pandan juga memiliki manfaat kesehatan seperti menyegarkan dan menghilangkan bau tidak sedap.",
        "benefits": ["Menyegarkan", "Menghilangkan bau tidak sedap"]
    },
    "Nangka": {
        "description": "Nangka adalah buah tropis yang sering digunakan dalam masakan dan kue. Buah nangka juga memiliki manfaat kesehatan seperti menyediakan serat untuk pencernaan yang sehat.",
        "benefits": ["Menyediakan serat", "Membantu pencernaan"]
    },
    "Lidah buaya": {
        "description": "Lidah buaya atau aloe vera adalah tanaman yang sering digunakan dalam produk perawatan kulit dan rambut. Lidah buaya juga memiliki manfaat kesehatan seperti menyembuhkan luka dan menyediakan kelembapan untuk kulit.",
        "benefits": ["Menyembuhkan luka", "Menyediakan kelembapan untuk kulit"]
    },
    "Kemangi": {
        "description": "Kemangi adalah tanaman yang memiliki aroma harum dan sering digunakan dalam masakan Indonesia. Kemangi juga dikenal memiliki manfaat kesehatan seperti meningkatkan nafsu makan dan mengatasi gangguan pencernaan ringan.",
        "benefits": ["Meningkatkan nafsu makan", "Mengatasi gangguan pencernaan ringan"]
    },
    "Jeruk Nipis": {
        "description": "Jeruk nipis adalah buah yang sering digunakan dalam minuman dan masakan. Buah jeruk nipis juga memiliki manfaat kesehatan seperti menyediakan vitamin C dan meningkatkan sistem kekebalan tubuh.",
        "benefits": ["Menyediakan vitamin C", "Meningkatkan sistem kekebalan tubuh"]
    },
    "Jambu Biji": {
        "description": "Jambu biji adalah buah tropis yang kaya akan vitamin C dan serat. Jambu biji juga memiliki manfaat kesehatan seperti membantu menjaga kesehatan kulit dan menjaga kesehatan pencernaan.",
        "benefits": ["Menjaga kesehatan kulit", "Menjaga kesehatan pencernaan"]
    }
}


# Fungsi untuk memproses gambar menggunakan model H5
def process_image(image_path):
    input_shape = model.input_shape[1:3]
    image = Image.open(image_path).resize(input_shape)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    return predictions

# Endpoint untuk memproses gambar dan melakukan klasifikasi
@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image = request.files['image']
        image_path = 'temp_image.jpg'  # Simpan sementara gambar

        image.save(image_path)
        predictions = process_image(image_path)
        
        labels = ["Seledri", "Jambu Biji", "Jeruk Nipis", "Kemangi", "Lidah Buaya", "Nangka", "Pandan", "Pepaya"]  # Ganti dengan label yang sesuai dengan model Anda
        results = [{'label': label, 'probability': float(prediction)} for label, prediction in zip(labels, predictions[0])]
        
        # Cari hasil untuk setiap tanaman
        plant_results = []
        max_prob = 0
        max_label = None
        for result in results:
            label = result['label']
            probability = result['probability']
            if label in plant_info and probability > max_prob:
                max_prob = probability
                max_label = label

        if max_label:
            description = plant_info[max_label]['description']
            benefits = plant_info[max_label]['benefits']
            plant_results.append({
                'label': max_label,
                'description': description,
                'benefits': benefits,
                'probability': max_prob
            })
        else:
            return jsonify({'error': 'No matching plant found'}), 404

        return jsonify({'plantResults': plant_results}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)
