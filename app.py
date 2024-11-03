from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Data RAB (id: 1)
rab_items = ["kursi", "meja", "pensil", "kertas", "buku", "penghapus", "spidol", "pen", "penggaris"]
rab_totals = [67000, 39000, 23000, 76000, 287000, 64000, 93800, 20934, 37400]

# Data laporan (id: 1)
laporan_items = ["kursi", "meja", "pensil", "kertas", "buku", "penghapus", "spidol", "pen", "penggaris"]
laporan_totals = [57000, 29000, 21000, 16000, 287000, 64000, 92000, 11000, 26400]

# Langkah 1: Tokenisasi item
all_items = [rab_items, laporan_items]

# Latih model Word2Vec pada daftar item
model = Word2Vec(all_items, min_count=1, vector_size=100, window=3, sg=0)

# Langkah 2: Hitung Cosine Similarity untuk setiap item
def get_avg_vector(item, model):
    vectors = [model.wv[word] for word in item if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

# Vektor rata-rata untuk setiap daftar item
rab_vector = get_avg_vector(rab_items, model)
laporan_vector = get_avg_vector(laporan_items, model)

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

total_similarity = compare_totals(rab_totals, laporan_totals)

# Langkah 4: Gabungkan kesamaan item dan kesamaan total
# Menggunakan weighted average (bobot bisa disesuaikan)
final_score = 0.5 * item_similarity + 0.5 * total_similarity

# Tampilkan hasil
print(f"Item Similarity: {item_similarity}")
print(f"Total Cost Similarity: {total_similarity}")
print(f"Final Similarity Score: {final_score}")
