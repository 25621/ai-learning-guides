# PPO untuk Kontrol Kontinu: Membuat BipedalWalker Berjalan

## Tindakan Diskret vs. Kontinu

Sejauh ini, setiap lingkungan yang kita selesaikan memiliki tindakan **diskret**:
- CartPole: dorong KIRI atau dorong KANAN (2 pilihan)
- LunarLander: tidak melakukan apa pun / kiri / utama / kanan (4 pilihan)

Tetapi robot di dunia nyata membutuhkan tindakan **kontinu**:
- Robot humanoid: "seberapa keras mendorong setiap sendi" (nilai apa pun dari -1 hingga +1)
- Mobil: "tepatnya seberapa banyak memutar setir" (sudut apa pun dari -30° hingga +30°)
- Lengan robot: "berikan gaya tepat 2,3 Newton ke arah ini"

**Contoh kehidupan nyata:** Mengetik di keyboard = diskret (tekan A, B, C...). Menulis dengan pensil = kontinu (gerakkan tangan 2,3 cm ke kanan, tekan dengan gaya 40g...).

---

## Kebijakan Gaussian untuk Tindakan Kontinu

Untuk tindakan kontinu, alih-alih distribusi Kategorikal (pilih dari N kategori), kita menggunakan **distribusi Normal (Gaussian)**:

```
Tindakan ~ Normal(μ, σ)
```

Di mana:
- **μ (mu, mean)**: Pusat distribusi — nilai tindakan yang "dibidik" oleh jaringan
- **σ (sigma, standar deviasi)**: Penyebaran — seberapa banyak keacakan / eksplorasi yang ditambahkan

```
        Probabilitas
             │
        0,4 ─┤      ██████
             │    ████████████
        0,2 ─┤  ██████████████████
             │████████████████████████
             └──────────────────────── Nilai tindakan
           -1  -0,5   0   0,5   1
                      ↑
                   mean μ
```

**Contoh kehidupan nyata:** Seorang pemanah ahli membidik pusat sasaran (μ). Anak panah mereka tidak semuanya mendarat di titik yang sama persis — ada beberapa penyebaran (σ). Saat mereka berlatih, mereka menjadi lebih akurat (σ berkurang) sambil tetap berpusat pada titik tengah.

---

## Jaringan Actor-Critic Gaussian Kami

```
Status (24 angka) → [256 neuron] → [256 neuron] →
    ├── Aktor: 4 nilai mean (μ₁, μ₂, μ₃, μ₄)
    │          + 4 parameter log_std (berbagi di semua status!)
    └── Kritikus: 1 nilai (V(s))
```

`log_std` (logaritma dari **standar deviasi** — ukuran penyebaran atau ketidakpastian) adalah **parameter yang dapat dipelajari** — tidak bergantung pada status. Ini menjaganya tetap sederhana sambil tetap membiarkan eksplorasi berubah selama pelatihan.

**Mengapa log_std alih-alih std?** Standar deviasi harus positif. Menggunakan `log_std` memungkinkan jaringan mengeluarkan angka riil apa pun (positif atau negatif), lalu kita menerapkan `exp(log_std)` — fungsi eksponensial, yang merupakan kebalikan dari logaritma — untuk mendapatkan std yang dijamin positif. Ini mencegah std menjadi negatif atau nol.

---

## Menghitung Log Probabilitas untuk Tindakan Kontinu

Untuk tindakan diskret: `log_prob = log(P(tindakan=KIRI))`

Untuk tindakan kontinu, **distribusi Normal** mendeskripsikan kurva berbentuk lonceng yang halus di sekitar mean. Sebuah nilai tepat tunggal memiliki probabilitas nol dalam matematika kontinu, jadi kita menggunakan tinggi kurva pada nilai tersebut, yang disebut **pdf** (probability density function):
```
log_prob = Σᵢ log[Normal(μᵢ, σᵢ).pdf(aᵢ)]
```

`log` berarti logaritma natural. Ia mengubah nilai densitas yang sangat kecil menjadi angka stabil yang lebih mudah dioptimalkan oleh jaringan saraf. Kita menjumlahkan di semua dimensi tindakan (4 untuk BipedalWalker), karena tindakan penuh adalah satu vektor berisi 4 angka.

**Contoh kehidupan nyata:** Berapa probabilitas suhu besok tepat 5,732...°C? Untuk cuaca kontinu, Anda akan melihat kurva distribusi Normal dan melihat seberapa tinggi kurva tersebut pada titik tepat tersebut. Suhu yang lebih mungkin (dekat dengan mean) memiliki probabilitas lebih tinggi.

---

## BipedalWalker: Tantangan Berjalan

BipedalWalker-v3 adalah robot 2D yang harus belajar berjalan tanpa jatuh:

```
          O (kepala)
         /│\
        / │ \
       /  │  \
      L   │   R   ← dua kaki, masing-masing dengan sendi lutut
     / \  │  / \
    ●   ● │ ●   ●  ← 4 motor (panggul/lutut untuk setiap kaki)
```

**Ruang status (24 angka):**
- Hull: sudut, kecepatan sudut, kecepatan horizontal, kecepatan vertikal (4 angka)
- Sendi: 4 motor (2 panggul, 2 lutut) masing-masing menyediakan sudut dan kecepatan, ditambah 2 sensor kontak tanah (satu untuk setiap kaki) (10 angka)
- 10 sensor jarak LIDAR (pembacaan jarak yang melihat tanah di depan) (10 angka)

**Ruang tindakan (4 nilai kontinu, masing-masing dalam [-1, 1]):**
Nilai tindakan mengontrol **torsi** (gaya rotasi yang diterapkan oleh motor) untuk tepat 4 sendi (tidak ada tindakan yang diterapkan langsung ke hull):
- Torsi panggul kaki 1, Torsi lutut kaki 1, Torsi panggul kaki 2, Torsi lutut kaki 2

**Imbalan:**
- +300 karena mencapai tujuan (sisi kanan)
- -100 karena jatuh (menyentuh tanah dengan badan)
- Imbalan kecil per langkah kemajuan ke depan
- Penalti kecil untuk setiap penggunaan mesin (efisiensi imbalan)

**Selesai ketika:** Rata-rata imbalan > 300 selama 100 episode

---

## Perbedaan Utama dari PPO Diskret

Semuanya sama KECUALI:

| | PPO Diskret | PPO Kontinu |
|---|---|---|
| **Kebijakan** | Kategorikal(logits) | Normal(μ, σ) |
| **Sampel** | tindakan = sampel dari {0,1,...,N} | tindakan = μ + σ × noise |
| **log_prob** | log P(tindakan=k) | Σ log Normal(μᵢ, σᵢ).pdf(aᵢ) |
| **Clamp** | Tidak diperlukan | Clamp tindakan ke [-1, 1] |

**Logits** adalah skor mentah yang belum dinormalisasi untuk tindakan diskret. Kebijakan kategorikal mengubahnya menjadi probabilitas dengan **softmax** — fungsi yang mengambil kumpulan angka apa pun dan menekannya menjadi distribusi probabilitas yang valid (semua nilai positif, berjumlah 1). Sebagai contoh, logits [2.0, 1.0, 0.5] menjadi probabilitas [0.59, 0.24, 0.17]. PPO kontinu **tidak** menggunakan softmax untuk tindakan itu sendiri, karena tindakan tidak dipilih dari menu tetap. Sebaliknya, kebijakan mengeluarkan mean dan standar deviasi dari distribusi Normal, lalu mengambil sampel torsi bernilai riil darinya.

**Clamp** berarti memaksa suatu nilai ke dalam rentang yang valid. Kode menggunakan `action.clamp(-1, 1)` sehingga lingkungan tidak pernah menerima perintah motor di luar batas yang diperbolehkan.

**Clip** dalam PPO berarti sesuatu yang berbeda: PPO memotong (clip) rasio probabilitas di dalam loss, seperti yang dijelaskan dalam [bagian clipping PPO](./ppo_scratch_explained.md#the-clipping-trick). Clamping tindakan melindungi antarmuka lingkungan; clipping PPO melindungi pembaruan kebijakan.

---

## Berjalan dari Nol: Apa yang Dipelajari Agen

**Pelatihan awal (imbalan negatif):** Robot meronta secara acak, langsung jatuh. Setiap episode berakhir dengan tabrakan dalam hitungan detik.

**Pelatihan menengah:** Robot menemukan bahwa menggerakkan kaki secara bergantian menciptakan kemajuan ke depan. Ia mulai melakukan langkah-langkah kecil yang canggung — imbalan menjadi kurang negatif.

**Pelatihan akhir:** Gaya berjalan (**gait**) yang mulus dan efisien muncul. Gait adalah pola gerakan yang berulang, seperti langkah kiri dan kanan yang bergantian. Robot menyesuaikan diri dengan medan yang tidak rata secara dinamis dengan memanfaatkan sensor LIDAR-nya untuk mengadaptasi langkahnya secara real-time.

**Contoh kehidupan nyata:** Bayi yang belajar berjalan:
1. Langsung jatuh (imbalan negatif)
2. Mengambil satu langkah, jatuh (sedikit kurang negatif)
3. Mengambil beberapa langkah (imbalan positif kecil)
4. Berjalan melintasi ruangan (imbalan positif besar!)

---

## Mengapa BipedalWalker Membutuhkan PPO (Bukan REINFORCE)

- **Episode BipedalWalker** bisa mencapai 1600 langkah (jauh lebih lama dari CartPole!)
- **Imbalan bersifat jarang** — imbalan kemajuan ke depan sangat kecil per langkah
- **REINFORCE akan membutuhkan** ribuan episode lengkap untuk mendapatkan sinyal yang berguna

Pemberbaruan n-step PPO dengan [GAE (Generalized Advantage Estimation)](./ppo_scratch_explained.md#gae-smarter-advantage-estimates) membiarkan robot belajar dari episode yang belum selesai:
> "Meskipun saya jatuh setelah 50 langkah, langkah-langkah tersebut menunjukkan BEBERAPA kemajuan ke depan. Biarkan saya menggunakan estimasi return 50-langkah daripada menunggu penyelesaian episode."

---

## Hasil

Setelah 500 pembaruan (≈ 1 juta langkah lingkungan):
- Robot menunjukkan kemajuan yang terlihat dari meronta secara acak menuju beberapa gerakan ke depan
- Peningkatan yang konsisten dalam kurva pembelajaran
- Konvergensi penuh ke imbalan > 300 membutuhkan lebih banyak pelatihan (5-10 juta langkah)

Kurva pembelajaran menunjukkan karakteristik "kurva-S" dari kontrol kontinu:
1. Kemajuan awal yang lambat (stabilitas pembelajaran)
2. Peningkatan cepat (penemuan gaya berjalan)
3. Penyempurnaan bertahap (optimalisasi gaya berjalan)

---

## Poin-Poin Penting

| Konsep | Bahasa Sederhana |
|---------|---------------|
| **Kebijakan Gaussian** | Alih-alih memilih dari menu, lempar dart pada rentang nilai |
| **μ (mean)** | Di mana kebijakan "membidik" |
| **σ (std)** | Seberapa banyak keacakan / eksplorasi yang digunakan kebijakan |
| **log_std sebagai parameter terpelajari** | Tingkat eksplorasi global yang diperbarui oleh optimasi berbasis gradien (gradient *ascent* pada imbalan, atau setara dengan gradient *descent* pada loss PPO) — persis seperti bobot jaringan lainnya |
| **Kontrol kontinu** | Mengontrol output bernilai riil (torsi, gaya, sudut) |

---

## Apa Selanjutnya?

PPO memiliki banyak **hiperparameter** — pengaturan yang Anda pilih sebelum pelatihan dimulai (berbeda dengan *parameter* seperti bobot jaringan, yang dipelajari secara otomatis). Contohnya termasuk `clip_eps`, learning rate, jumlah epoch, dan batch size.

Seberapa sensitif PPO terhadap pilihan-pilihan ini? `ppo_hyperparams.py` menjalankan eksperimen dengan memvariasikan setiap hiperparameter secara sistematis dan menunjukkan efeknya pada kecepatan dan stabilitas pembelajaran.
