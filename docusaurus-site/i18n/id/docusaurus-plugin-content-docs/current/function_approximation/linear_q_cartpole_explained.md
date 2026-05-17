# Q-Learning Linear untuk CartPole 🎪

## Apa Itu CartPole?

Bayangkan sebuah gagang sapu yang diseimbangkan tegak di atas jari Anda. Jika Anda menggerakkan jari ke kiri atau ke kanan sedikit saja, Anda dapat menjaga gagang sapu agar tidak jatuh. Itulah **CartPole**!

Sebuah robot kecil duduk di atas kereta (kotak beroda) dan memiliki tiang yang menonjol di atasnya. Robot hanya bisa mendorong kereta ke **kiri** atau ke **kanan**. Ia harus belajar menjaga agar tiang tetap seimbang selama mungkin — persis seperti Anda menyeimbangkan gagang sapu!

Robot dapat melihat 4 hal tentang dunianya:
1. Di mana posisi kereta berada
2. Seberapa cepat kereta bergerak
3. Seberapa miring tiang tersebut
4. Seberapa cepat tiang itu miring

---

## Masalah Besar: Terlalu Banyak Status!

Ingat Q-learning dari Fase 2? Ia menggunakan tabel besar untuk mengingat seberapa baik setiap tindakan dalam setiap situasi (status). Itu berhasil dengan baik untuk Frozen Lake — hanya ada 16 kotak di atas es.

Tetapi CartPole berbeda! Kereta bisa berada di **posisi mana pun**, bergerak dengan **kecepatan mana pun**, dengan tiang pada **sudut mana pun**. Pada dasarnya ada **status yang mungkin tak terbatas**! Kita tidak bisa membuat tabel dengan baris yang tak terbatas. Kita akan membutuhkan buku catatan seukuran alam semesta!

**Contoh kehidupan nyata:** Bayangkan Anda sedang belajar naik sepeda. Anda tidak bisa menghafal setiap kemungkinan goyangan — jumlahnya terlalu banyak! Sebaliknya, Anda mempelajari sebuah **aturan**: "saat saya miring ke kiri, dorong ke kanan; saat saya miring ke kanan, dorong ke kiri." Aturan sederhana berlaku untuk SEMUA goyangan.

---

## Solusi: Rumus Ajaib

**Aproksimasi fungsi linear (Linear function approximation)** menggantikan tabel raksasa dengan sebuah **rumus kecil**:

> **Skor(situasi, tindakan) = w₁ × posisi_kereta + w₂ × kecepatan_kereta + w₃ × sudut_tiang + w₄ × kecepatan_tiang**

- Angka `w` disebut **bobot (weight)** — mereka seperti tombol yang bisa Anda putar
- Kita mempelajari **bobot yang berbeda untuk setiap tindakan** (dorong-kiri dan dorong-kanan)
- Rumus tersebut memberikan skor tentang seberapa baik setiap tindakan saat ini

**Contoh kehidupan nyata:** Pikirkan resep sederhana: "1 cangkir tepung + 2 telur + ½ cangkir mentega." Bobot (1, 2, ½) memberi tahu Anda seberapa penting setiap bahan. Kita sedang mempelajari resep untuk keputusan yang baik!

---

## Bagaimana Cara Belajarnya?

Robot mencoba berbagai hal, mendapatkan umpan balik, dan menyesuaikan bobot:

1. **Robot mendorong kereta** (memilih tindakan dengan skor tertinggi)
2. **Fisika terjadi** (tiang miring sedikit, kereta bergerak)
3. **Robot mendapatkan imbalan** (+1 untuk setiap langkah tiang tetap tegak, 0 jika jatuh)
4. **Robot bertanya:** "Apakah hasil aktualnya lebih baik atau lebih buruk daripada yang saya prediksi?"
5. **Robot menyesuaikan bobot** agar lebih dekat dengan kenyataan di lain waktu

Ini adalah **Semi-Gradient TD Update** — nama keren untuk "menyesuaikan resep sedikit berdasarkan kejutan."

> **Bobot baru = Bobot lama + Tingkat pembelajaran × (Apa yang sebenarnya terjadi − Apa yang saya prediksi) × Fitur**

---

## Apa yang Ditemukan Kode Kami

Saat Anda menjalankan `linear_q_cartpole.py`, robot:

- Dimulai dengan sangat buruk (tiang jatuh dalam 10–30 langkah)
- Secara bertahap mempelajari bobot yang baik selama 3.000 percobaan
- Akhirnya menjaga tiang tetap seimbang selama 100–400+ langkah!

Plot tersebut menunjukkan **kurva pembelajaran** — bagaimana skor menjadi lebih baik seiring waktu. Ini akan bergelombang (belajar tidak pernah mulus!), tetapi trennya harus naik.

---

## Mengapa Ini Keren (dan Terbatas!)

**Keren:** Rumus kecil dengan hanya 8 angka (4 bobot × 2 tindakan) dapat menyeimbangkan tiang! Tidak perlu tabel raksasa.

**Terbatas:** Rumusnya terlalu sederhana untuk tugas yang kompleks. Ia mengasumsikan angka yang lebih besar selalu berarti efek yang lebih besar (yang tidak selalu benar). Untuk game yang lebih sulit seperti Atari, kita membutuhkan **jaringan saraf (neural networks)** — yang dilakukan oleh DQN!

---

## Kosakata Kunci

| Kata | Makna |
|------|---------|
| **Fitur (Feature)** | Satu hal yang dapat diukur tentang dunia (misalnya, sudut tiang) |
| **Bobot (Weight)** | Seberapa besar suatu fitur memengaruhi keputusan |
| **Linear** | Rumusnya hanya perkalian dan penjumlahan (tidak ada kurva yang rumit) |
| **Semi-gradient** | Memperbarui bobot dengan mengikuti arah kesalahan yang lebih sedikit |
| **Aproksimasi fungsi** | Menggunakan rumus alih-alih tabel |

---

## Apa Selanjutnya?

Aproksimasi linear seperti menggunakan penggaris lurus untuk menggambar kurva — ia bekerja oke untuk bentuk sederhana tetapi tidak untuk yang kompleks. Untuk game Atari dengan jutaan kemungkinan situasi, kita membutuhkan **Deep Q-Networks (DQN)** — jaringan saraf yang dapat mempelajari pola yang jauh lebih kompleks. Itu ada di file berikutnya!
