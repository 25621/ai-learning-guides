# REINFORCE: Mengajari Robot Membuat Pilihan Lebih Baik

## Apa yang Sedang Kita Coba Lakukan?

Bayangkan Anda memiliki robot yang sedang memainkan video game. Setiap detik, robot tersebut harus memilih:
**"Haruskah saya menekan tombol atau tidak?"**

Alih-alih menghafal setiap situasi dalam sebuah tabel (seperti Q-learning), kita ingin robot mempelajari sebuah **resep** — sekumpulan aturan yang secara langsung mengatakan: "Dalam situasi ini, lakukan tindakan ini."

Resep ini disebut **kebijakan (policy)** (π, pi). Dalam pembelajaran penguatan (reinforcement learning), π berarti "aturan untuk memilih tindakan."

---

## Cara Lama vs. Cara Baru {#the-old-way-vs-the-new-way}

**Cara Lama (Q-learning / DQN):** Pelajari seberapa BAGUS setiap tindakan (nilai-Q), lalu pilih yang terbaik.
> "Mendorong KIRI memiliki skor 7, mendorong KANAN memiliki skor 5 → dorong KIRI!"

**Cara Baru (Policy Gradient):** Pelajari secara langsung tindakan mana yang harus DIPILIH.
> "Ketika tiang miring ke kanan, dorong KANAN dengan peluang 80%, dorong KIRI dengan peluang 20%."
*(Kata **Gradien** mengacu pada "langkah" matematis yang kita ambil untuk menyesuaikan probabilitas ini secara perlahan ke arah yang benar.)*

**Contoh kehidupan nyata:** Belajar naik sepeda.
- Cara lama: menghitung *skor tepat* untuk "miring ke kiri 5 derajat" vs "miring ke kiri 7 derajat."
- Cara baru: cukup berlatih sampai **tubuh** Anda belajar — dorong kaki yang terasa benar!

---

## Bagaimana Cara Kerja REINFORCE?

REINFORCE menonton robot memainkan permainan penuh dari awal sampai akhir (satu **episode**), lalu bertanya: "Tindakan mana yang menghasilkan skor bagus? Mari lakukan lebih banyak tindakan itu!"

### Langkah demi Langkah

**1. Mainkan satu episode**

Robot membuat pilihan dan mengumpulkan pengalaman:
```
Langkah 1: Status = [tiang miring ke kanan] → Tindakan = dorong KANAN → Imbalan = +1
Langkah 2: Status = [tiang hampir seimbang] → Tindakan = dorong KANAN → Imbalan = +1
Langkah 3: Status = [tiang miring ke kiri] → Tindakan = dorong KIRI → Imbalan = +1
...
Langkah 47: Status = [tiang jatuh!] → Episode berakhir
```

**2. Hitung return**

Untuk setiap langkah, hitung G_t — **total imbalan dari titik tersebut ke depan**:
```
G pada langkah 47 = 1
G pada langkah 46 = 1 + 0,99 × 1 = 1,99
G pada langkah 45 = 1 + 0,99 × 1,99 = 2,97
...
G pada langkah 1 = 47 (kira-kira — return lebih tinggi karena dari awal)
```

γ = 0,99 **faktor diskon** berarti imbalan di masa depan yang jauh dihitung sedikit lebih rendah.

**Contoh kehidupan nyata:** Mendapatkan bintang emas di hari pertama sekolah terasa lebih menyenangkan daripada mengetahui Anda *mungkin* mendapatkannya di hari ke-100. Imbalan masa depan "didiskon" sedikit.

**3. Perbarui kebijakan**

Untuk setiap tindakan yang diambil:
> Jika G_t TINGGI (tindakan itu menghasilkan hasil yang luar biasa): **lakukan lebih sering!**
> Jika G_t RENDAH (tindakan itu menghasilkan hasil yang buruk): **lakukan lebih jarang!**

Matematikanya: `loss = -log_prob(action) × G_t`

Mengambil gradien dan memperbarui kebijakan itu seperti memberi tahu robot:
*"Tindakan yang kamu ambil di langkah 20? Kamu harus melakukannya 5% lebih sering lain kali!"*

---

## Apa Itu Jaringan Kebijakan (Policy Network)?

Alih-alih tabel, kita menggunakan **jaringan saraf** untuk merepresentasikan kebijakan.

```
Observasi        Jaringan Kebijakan     Probabilitas Tindakan
[posisi krt] →  [128 neuron] →         [dorong KIRI: 30%]
[kecep krt]  →  [128 neuron]           [dorong KANAN: 70%]
[sudut tng]  →
[kecep tng]  →
```

Jaringan tersebut mengeluarkan **probabilitas** untuk setiap tindakan. Kita kemudian mengambil sampel:
> Lempar dadu → 1-30: dorong KIRI, 31-100: dorong KANAN

**Contoh kehidupan nyata:** Aplikasi cuaca mengatakan "70% peluang hujan." Anda tidak TAHU akan hujan — Anda memutuskan berdasarkan probabilitas. Robot melakukan hal yang sama!

---

## Normalisasi: Mengapa Kita Mengurangi Rata-rata

Sebelum menggunakan G_t untuk memperbarui, kita menormalisasi:
```
G_dinormalisasi = (G - rata_rata(G)) / standar_deviasi(G)
```

**Mengapa?** Bayangkan semua imbalan bernilai positif (seperti pada CartPole — selalu +1 per langkah). Tanpa normalisasi, SETIAP tindakan terlihat "bagus" dan sinyal pembaruannya membingungkan.

Setelah normalisasi, beberapa return menjadi positif (di atas rata-rata → dorong lebih banyak), dan beberapa menjadi negatif (di bawah rata-rata → dorong lebih sedikit). Sinyal tersebut menjadi jauh lebih bersih!

**Contoh kehidupan nyata:** Guru Anda memberikan nilai berdasarkan kurva. Jika skor rata-rata adalah 70 dan Anda mendapat 85, itu bagus! Tetapi jika rata-rata adalah 90 dan Anda mendapat 85, itu di bawah rata-rata. Skor mentah saja tidak menceritakan keseluruhan cerita.

---

## Masalahnya: Variansi Tinggi

REINFORCE memiliki satu kelemahan besar: **variansi**. Return G_t sangat berisik.

**Contoh kehidupan nyata:** Bayangkan menilai seorang koki hanya dengan mencicipi SATU hidangan dari setiap restoran. Terkadang koki tersebut sedang mengalami hari yang buruk, terkadang bahan-bahannya sedang kurang bagus. Satu hidangan tidak cukup untuk mengetahui secara andal apakah restoran tersebut bagus!

REINFORCE menunggu satu episode PENUH sebelum melakukan pembaruan. Satu episode mungkin sangat beruntung, episode lainnya sangat tidak beruntung. Gradiennya melompat ke mana-mana.

Inilah sebabnya mengapa kurva pembelajaran (dalam plot) terlihat bergerigi:
- Beberapa kali berjalan mencapai 500 (luar biasa!)
- Beberapa kali turun ke 50 (buruk!)

Meskipun berisik, REINFORCE akhirnya belajar — tetapi membutuhkan kesabaran.

---

## Hasil

```
Episode 100 | Rata-rata imbalan (100 terakhir): 43,1
Episode 200 | Rata-rata imbalan (100 terakhir): 193,9
Episode 500 | Rata-rata imbalan (100 terakhir): 408,4
Episode 1000 | Rata-rata imbalan (100 terakhir): 500,0 ← Selesai!
```

Robot belajar menyeimbangkan tiang untuk maksimum 500 langkah — SELESAI!

Meskipun memiliki masalah variansi, REINFORCE pada CartPole efektif karena:
1. Episodenya singkat (sehingga kita mendapatkan banyak episode per pelatihan)
2. Kebijakan optimalnya sederhana (sebagian besar mendorong ke arah tiang miring)

---

## Poin-Poin Penting

| Konsep | Bahasa Sederhana |
|---------|---------------|
| **Kebijakan** | Resep robot untuk memilih tindakan |
| **Episode** | Satu permainan penuh dari awal sampai akhir |
| **Return G_t** | Total imbalan masa depan dari saat ini |
| **Diskon γ** | Imbalan masa depan sedikit kurang berharga dibandingkan imbalan segera |
| **Normalisasi** | Mengurangi rata-rata agar sinyal lebih bersih |
| **Variansi** | Seberapa banyak estimasi gradien melompat-lompat |

---

## Apa Selanjutnya?

Kelemahan besar REINFORCE adalah **variansi**. Dalam skrip berikutnya (`reinforce_baseline.py`), kita menambahkan sebuah **baseline** yang secara dramatis mengurangi kebisingan ini — tanpa mengubah apa yang dipelajari algoritma secara rata-rata.

Ide kuncinya: alih-alih bertanya "apakah tindakan ini bagus?", kita bertanya "apakah tindakan ini **lebih baik dari yang diharapkan**?" Perubahan kecil itu membuat pembelajaran jauh lebih stabil.
