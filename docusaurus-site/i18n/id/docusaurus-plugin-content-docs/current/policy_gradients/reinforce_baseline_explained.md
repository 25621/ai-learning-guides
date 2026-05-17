# REINFORCE dengan Baseline: Menembus Kebisingan

## Masalah pada REINFORCE Biasa

Bayangkan Anda adalah seorang siswa yang mencoba memutuskan apakah jawaban Anda dalam sebuah ujian itu bagus.

**Umpan balik yang buruk:** "Kamu mendapat 7 poin!"

Apakah 7 itu bagus? Jika maksimumnya 10, ya! Jika orang lain mendapat 9, tidak! Tanpa konteks, Anda tidak bisa tahu apakah Anda harus mengubah gaya jawaban Anda.

Inilah tepatnya masalah pada REINFORCE: ia menggunakan **return mentah** (G_t) untuk mengevaluasi tindakan. Skor total return 200 poin mungkin luar biasa atau mengerikan tergantung pada situasinya.

---

## Menambahkan Baseline

Sebuah **baseline** b(s) adalah titik acuan: "Imbalan apa yang saya **harapkan** dalam situasi ini?"

Alih-alih bertanya "Apakah tindakan ini bagus?", kita bertanya:

> **"Apakah tindakan ini lebih baik dari apa yang biasanya saya harapkan?"**

```
Sinyal lama: pembaruan ∝ G_t
Sinyal baru: pembaruan ∝ (G_t - b(s_t))
```

**Contoh kehidupan nyata:** Anda mendapat nilai 85 pada ujian matematika.
- Jika rata-rata kelas adalah 60 → jawaban Anda **25 poin di atas rata-rata** → luar biasa!
- Jika rata-rata kelas adalah 90 → jawaban Anda **5 poin di bawah rata-rata** → perlu perbaikan!

**Advantage** (G_t - b(s)) bernilai positif ketika Anda melakukan lebih baik dari yang diharapkan dan negatif ketika Anda melakukan lebih buruk. Ini adalah sinyal pembelajaran yang jauh lebih bersih!

---

## Apa Itu Baseline?

Baseline yang alami adalah **fungsi nilai V(s)**:

> V(s) = "Total imbalan yang diharapkan jika saya berada di status s dan mengikuti kebijakan saya saat ini"

Kita mempelajari ini dengan **Jaringan Nilai (Value Network)** terpisah (juga disebut jaringan baseline atau kritikus):

```
Status → [128 neuron] → [128 neuron] → V(s) (angka tunggal)
```

Untuk setiap status yang dikunjungi agen, V(s) memprediksi return yang diharapkan. Jika return aktual G_t lebih tinggi dari V(s), tindakan tersebut lebih baik dari yang diharapkan!

---

## Dua Jaringan Belajar Bersama

```
Episode terjadi
     ↓
Hitung return aktual G_t
     ↓
         ┌─────────────────────────────┐
         │ Advantage = G_t - V(s_t)    │
         │ +: tindakan lebih baik      │
         │ -: tindakan lebih buruk     │
         └─────────────────────────────┘
              ↓                  ↓
   Perbarui Jaringan Kebijakan  Perbarui Jaringan Nilai
   (jadikan tindakan baik lebih (jadikan prediksi lebih
   /kurang mungkin terjadi)     akurat lain kali)
```

**Contoh kehidupan nyata:** Dua teman pergi ke restoran bersama.

- Teman 1 (Jaringan Nilai): "Saya prediksi hidangan ini akan bernilai 7/10"
- Teman 2 (Jaringan Kebijakan): Anda mencoba hidangan tersebut dan memberinya nilai 9/10
- Advantage = 9 - 7 = +2 → "Itu lebih baik dari yang diharapkan! Pesan lagi!"

Kunjungan berikutnya, Teman 1 memperbarui prediksinya lebih dekat ke 9/10. Teman 2 lebih mungkin memesan hidangan itu lagi.

---

## Mengapa Ini Mengurangi Variansi?

**Bukti matematis (intuisi):**

Tanpa baseline: `gradien ∝ ∇log π(a|s) × G_t`

Nilai G_t sangat bervariasi dari episode ke episode:
```
Episode 1: G = [45, 44, 43, ..., 1]   (permainan menengah)
Episode 2: G = [500, 499, ..., 1]      (permainan luar biasa!)
Episode 3: G = [12, 11, ..., 1]        (permainan buruk)
```

Estimasi gradien melompat secara liar karena G_t besar dan berisik.

Dengan baseline: `gradien ∝ ∇log π(a|s) × (G_t - V(s_t))`

Advantage (G_t - V(s_t)) jauh lebih kecil dan berpusat di dekat nol:
```
Episode 1: advantage ≈ [-2, +1, -3, ..., 0]   (kecil, terpusat)
Episode 2: advantage ≈ [+10, +8, ..., +3]      (permainan ini MEMANG bagus)
Episode 3: advantage ≈ [-5, -6, ..., -2]       (permainan ini MEMANG buruk)
```

**Contoh kehidupan nyata:** Mengukur kecepatan lari Anda.
- Tanpa baseline: "Saya berlari 8 km/jam" (tidak berarti tanpa konteks)
- Dengan baseline: "Saya berlari 2 km/jam LEBIH CEPAT dari rata-rata saya" (jelas bagus!)

Advantage selalu berupa perbandingan — secara alami lebih kecil dan lebih stabil.

---

## Yang Penting: Tidak Ada Bias!

Baseline tidak mengubah APA yang dipelajari algoritma — hanya SEBERAPA CEPAT dan STABIL ia belajar.

**Mengapa?** Karena advantage yang diharapkan selalu 0 dalam ekspektasi:

> E[G_t - V(s_t)] = E[G_t] - V(s_t) = V(s_t) - V(s_t) = 0

Setiap b(s) yang tidak bergantung pada tindakan berfungsi sebagai baseline yang valid!

**Contoh kehidupan nyata:** Penilaian berdasarkan kurva tidak mengubah siapa yang berprestasi terbaik — itu hanya membuat skor lebih mudah diinterpretasikan. Peringkatnya tetap sama; hanya skalanya yang berubah.

---

## Hasil

```
Tanpa baseline — Rata-rata 100-ep akhir: 500.0, grad var: 599.3
Dengan baseline — Rata-rata 100-ep akhir: 491.4, grad var: 578.8
```

Kedua metode mencapai performa yang hampir sempurna pada CartPole, tetapi perhatikan:
1. **Variansi gradien** dapat diukur (plot sisi kanan menunjukkan variansi selama pelatihan)
2. Dengan baseline, agen mencapai performa tinggi **dengan lebih andal** — lebih sedikit penurunan kembali ke imbalan rendah selama pelatihan

Pengurangan variansi lebih dramatis di lingkungan yang lebih sulit (LunarLander, MuJoCo).

---

## Persamaan Utama

```
Nilai baseline:  V(s) ← V(s) + α(G_t - V(s))   [minimalkan MSE]
Gradien kebijakan: θ ← θ + α ∇log π(a_t|s_t) · (G_t - V(s_t))
Advantage:         A_t = G_t - V(s_t)
```

---

## Poin-Poin Penting

| Konsep | Bahasa Sederhana |
|---------|---------------|
| **Baseline b(s)** | Imbalan yang diharapkan di status s — titik acuan kita |
| **Advantage A_t** | "Apakah tindakan ini lebih baik dari yang diharapkan?" |
| **Value Network** | Jaringan saraf yang belajar memprediksi return yang diharapkan |
| **Pengurangan variansi** | Mengurangi kebisingan dalam estimasi gradien → pembelajaran lebih stabil |
| **Tidak bias (Unbiased)** | Baseline tidak mengubah target kebijakan rata-rata; ia hanya membuat sinyal pembelajaran kurang berisik dan lebih stabil |

---

## Apa Selanjutnya?

Baseline sebenarnya adalah awal dari sesuatu yang jauh lebih kuat: metode **Actor-Critic**.

Alih-alih menghitung V(s) hanya di akhir episode, Actor-Critic memperbarui V(s) di setiap langkah menggunakan pembelajaran **Temporal Difference**. Ini membuat pembaruan jauh lebih cepat dan memungkinkan agen untuk belajar dari episode yang belum selesai!

Lihat `a2c_lunarlander.py` untuk implementasi Actor-Critic yang lengkap.
