from flask import Flask, request, jsonify
import json
from model import calculate_similarity
import numpy as np

app = Flask(__name__)

def convert_to_serializable(result):
    """ Fungsi untuk mengonversi semua nilai numpy dan dict dalam dict ke tipe Python standar """
    for key, value in result.items():
        if isinstance(value, np.generic):
            result[key] = value.item()  # Konversi numpy ke tipe Python standar
        elif isinstance(value, dict):  # Handle nested dictionary
            result[key] = json.dumps(value)  # Konversi dict ke string JSON
    return result

# Fungsi untuk mendapatkan RAB berdasarkan ID
def get_rab_by_id(rab_id):
    with open('data/rab.json') as f:
        rab_list = json.load(f)
    # Cari data RAB dengan ID yang sesuai
    for rab in rab_list:
        if rab["id"] == rab_id:
            return rab
    return None

# Endpoint untuk menerima data laporan dan membandingkan dengan RAB berdasarkan rab_id
@app.route('/compare', methods=['POST'])
def compare_rab():
    # Data laporan dikirimkan sebagai JSON melalui body request
    laporan_data = request.json

    # Ambil ID RAB dari laporan (rab_id)
    rab_id = laporan_data.get('rab_id')

    # Ambil data RAB yang sesuai dengan rab_id
    rab_data = get_rab_by_id(rab_id)

    if not rab_data:
        return jsonify({"error": "RAB with id {} not found.".format(rab_id)}), 404

    # Hitung similarity antara data RAB dan laporan
    result = calculate_similarity(rab_data, laporan_data)

    result = convert_to_serializable(result)

    # Kembalikan hasil dalam format JSON
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
