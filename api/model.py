from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Fungsi untuk memproses data dan menghitung similarity
def calculate_similarity(rab_data, laporan_data):
    # Latih model Word2Vec pada daftar item
    model = Word2Vec([rab_data['items'], laporan_data['items']], min_count=1, vector_size=100, window=3, sg=0)

    # Langkah 2: Hitung Cosine Similarity untuk setiap item
    def get_avg_vector(item, model):
        vectors = [model.wv[word] for word in item if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(100)

    # Vektor rata-rata untuk setiap daftar item
    rab_vector = get_avg_vector(rab_data['items'], model)
    laporan_vector = get_avg_vector(laporan_data['items'], model)

    # Hitung kesamaan item menggunakan Cosine Similarity
    item_similarity = cosine_similarity([rab_vector], [laporan_vector])[0][0]

    # Langkah 3: Hitung perbedaan total biaya antara RAB dan laporan
    def compare_totals(rab_totals, laporan_totals):
        differences = []
        for rab, laporan in zip(rab_totals, laporan_totals):
            # Persentase perbedaan antara RAB dan laporan
            diff = abs(rab - laporan) / max(rab, laporan)
            differences.append(1 - diff)  # 1 berarti sangat mirip, 0 berarti sangat berbeda
        return np.mean(differences)

    total_similarity = compare_totals(rab_data['totals'], laporan_data['totals'])

    # Langkah 4: Gabungkan kesamaan item dan kesamaan total
    final_score = 0.5 * item_similarity + 0.5 * total_similarity
    
    # Pembulatan ke dua angka di belakang koma
    item_similarity = round(item_similarity, 2)
    total_similarity = round(total_similarity, 2)
    final_score = round(final_score, 2)
    print("error disini")

    return {
        "item_similarity": item_similarity,
        "total_similarity": total_similarity,
        "final_score": final_score
    }
