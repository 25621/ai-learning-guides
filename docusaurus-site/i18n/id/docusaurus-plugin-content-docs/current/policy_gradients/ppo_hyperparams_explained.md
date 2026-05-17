# Sensitivitas Hiperparameter PPO: Apa yang Paling Penting?

## Mengapa Hiperparameter Penting

Bayangkan memanggang kue cokelat. Resepnya membutuhkan:
- 2 telur
- 200g tepung
- 1 sendok teh baking powder
- 35 menit pada suhu 180°C

Jika Anda menggunakan 10 telur, kuenya meledak. Jika Anda menggunakan 0,1 sendok teh baking powder, kuenya tidak mengembang. Jika Anda memanggang pada suhu 300°C selama 10 menit, bagian luarnya gosong dan bagian dalamnya mentah.

**Hiperparameter dalam PPO seperti bahan-bahan dan pengaturan oven.** Kombinasi yang tepat bekerja dengan indah; pengaturan yang salah dapat mencegah pembelajaran sama sekali.

Skrip ini secara sistematis menguji 3 hiperparameter utama dengan hanya mengubah SATU parameter pada satu waktu, menjalankan setiap pengaturan dengan 3 seed acak yang berbeda, dan membandingkan hasilnya.

---

## Tiga Eksperimen

### Eksperimen 1: Clip Epsilon (ε)

```
ε = 0,05   (sangat konservatif — hanya perubahan kebijakan kecil yang diperbolehkan)
ε = 0,2    (standar — keamanan dan kecepatan yang seimbang)
ε = 0,4    (agresif — memungkinkan perubahan kebijakan besar)
```

**Apa yang dikontrol oleh ε?**

ε adalah ukuran "jendela keamanan" di sekitar kebijakan lama:
```
rasio harus tetap dalam [1 - ε,  1 + ε]
ε=0,05: rasio dalam [0,95, 1,05]  ← perubahan sangat kecil
ε=0,2:  rasio dalam [0,80, 1,20]  ← standar  
ε=0,4:  rasio dalam [0,60, 1,40]  ← perubahan besar
```

**Contoh kehidupan nyata:** Pikirkan ε sebagai "seberapa jauh Anda diperbolehkan memutar setir mobil dalam satu gerakan."
- ε=0,05: Seperti mengemudi di atas es — hanya penyesuaian sangat kecil
- ε=0,2:  Mengemudi normal — belokan yang wajar
- ε=0,4:  Pengemudi balap — penyetiran agresif, risiko **tergelincir (spinning out)** (kehilangan kendali karena perubahannya terlalu drastis, seperti mobil yang tergelincir keluar jalan)

**Hasil yang diharapkan:**
- ε=0,05: Pembelajaran lambat tetapi stabil (terlalu berhati-hati)
- ε=0,2:  Keseimbangan yang baik (nilai **"Goldilocks"** — tidak terlalu kecil, tidak terlalu besar, pas — dinamai menurut dongeng Goldilocks yang memilih bubur yang tidak terlalu panas maupun terlalu dingin)
- ε=0,4:  Dapat belajar cepat tetapi mungkin **melampaui target dan berosilasi (overshoot and oscillate)** (melampaui = melewati kebijakan optimal; berosilasi = memantul bolak-balik di sekitarnya tanpa menetap, seperti pendulum yang berayun terlalu jauh ke kedua arah)

---

### Eksperimen 2: Tingkat Pembelajaran (Learning Rate)

```
lr = 1e-4  (lambat tapi stabil)
lr = 3e-4  (standar)
lr = 1e-3  (cepat tapi berisiko)
```

**Apa yang dikontrol oleh tingkat pembelajaran?**

Learning rate seperti "ukuran langkah" saat mendaki bukit (setiap langkah = satu pembaruan pada bobot jaringan saraf, menggerakkannya sedikit ke arah yang meningkatkan imbalan):
- Terlalu kecil: Butuh waktu selamanya untuk mencapai puncak (konvergensi lambat)
- Terlalu besar: Anda melampaui puncak dan jatuh ke sisi lain (**divergen** — imbalan pelatihan runtuh atau berfluktuasi liar bukannya membaik dengan stabil)
- Pas: Kemajuan stabil menuju puncak

**Contoh kehidupan nyata:** Menyetel senar gitar.
- lr=1e-4: Putaran sangat kecil pada **pasak** penyetel (tombol yang Anda putar untuk mengencangkan atau mengendurkan senar) — butuh waktu lama tetapi presisi
- lr=3e-4: Penyetelan normal — menemukan nada yang tepat dalam beberapa putaran
- lr=1e-3: **Sentakan** besar (tarikan keras yang tiba-tiba) pada pasak — mungkin akan **memutuskan** senar (merusaknya sepenuhnya, sama seperti pembaruan yang terlalu besar dapat merusak pelatihan secara permanen)!

**Hasil yang diharapkan:**
- lr=1e-4: Akhirnya bagus tapi sangat lambat
- lr=3e-4: Performa terbaik secara keseluruhan
- lr=1e-3: Kemajuan awal yang cepat, lalu ketidakstabilan

---

### Eksperimen 3: Epoch Pembaruan (K)

```
K = 3   (konservatif — sedikit sapuan melalui setiap batch)
K = 10  (standar)
K = 20  (agresif — banyak sapuan melalui setiap batch)
```

**Apa yang dikontrol oleh epoch pembaruan?**

Setelah mengumpulkan **rollout** (= memainkan game untuk jangka waktu tertentu untuk mengumpulkan pengalaman baru — seperti siswa yang mengerjakan sesi PR sebelum meninjaunya), PPO mengemas pengalaman tersebut ke dalam sebuah **batch** (= set lengkap tupel status, tindakan, imbalan dari rollout tersebut). Ia kemudian menjalankan K **kali sapuan (passes)** (= sapuan penuh melalui batch, setiap sapuan memperbarui jaringan satu kali) atas data yang sama.
Lebih banyak epoch = memeras lebih banyak pembelajaran dari setiap batch, tetapi berisiko **overfitting pada data usang** (= menghafal pola yang benar di bawah kebijakan lama tetapi tidak lagi valid setelah kebijakan diperbarui, seperti siswa yang menghafal ujian tahun lalu dan gagal pada ujian baru).

**Contoh kehidupan nyata:** Seorang siswa berlatih dengan satu set 20 soal matematika.
- K=3:  Kerjakan setiap soal 3 kali → masih belajar, tidak overfit pada set latihan
- K=10: Kerjakan setiap soal 10 kali → penguasaan yang solid pada soal-soal spesifik ini
- K=20: Kerjakan setiap soal 20 kali → **menghafal solusi tanpa benar-benar memahami matematika** (= model sangat cocok dengan batch spesifik tersebut tetapi kehilangan kemampuan untuk menggeneralisasi)!

> ⚠️ **"Tetapi hasil untuk K=20 terlihat baik-baik saja — mengapa saya harus peduli?"**
> Trik clipping PPO membatasi seberapa banyak kebijakan dapat berubah per sapuan, jadi K=20 tidak akan menyebabkan keruntuhan mendadak.
> Namun, agen masih secara diam-diam beradaptasi secara berlebihan pada data yang tidak lagi mencerminkan apa yang sebenarnya akan dialami oleh kebijakan saat ini.
> Ini **memperlambat pembelajaran jangka panjang**: setiap rollout mengajarkan agen lebih sedikit dari yang seharusnya, karena sapuan selanjutnya mendaur ulang informasi yang semakin usang.
> Kerusakannya bertahap, tidak dramatis — itulah sebabnya mudah untuk diabaikan dalam eksperimen singkat.

Clipping mencegah overfitting katastropik, tetapi terlalu banyak epoch masih dapat memperlambat pembelajaran secara keseluruhan.

**Hasil yang diharapkan:**
- K=3:  Kurang efisien (beberapa potensi pembelajaran terbuang per batch)
- K=10: Keseimbangan yang baik
- K=20: Risiko kebijakan menjadi **terlalu percaya diri pada data usang** (= pembaruan jaringan didorong oleh pengalaman yang tidak lagi cocok dengan apa yang akan ditemui kebijakan saat ini, secara diam-diam mengikis efisiensi sampel)

---

## Cara Membaca Hasil

Plot menunjukkan tiga grafik, masing-masing memvariasikan satu hiperparameter:

```
Grafik kiri:   Clip Epsilon — ε mana yang belajar paling cepat?
Grafik tengah: Learning Rate — lr mana yang paling stabil?
Grafik kanan:  Update Epochs — K mana yang menemukan kebijakan terbaik?
```

Setiap baris adalah **rata-rata imbalan di 3 seed** (untuk mengurangi keacakan).

**Apa yang harus dicari:**
1. **Kecepatan belajar:** Baris mana yang mencapai imbalan tinggi paling cepat?
2. **Performa akhir:** Baris mana yang mencapai imbalan akhir tertinggi?
3. **Stabilitas:** Baris mana yang osilasinya paling sedikit?

Hiperparameter yang baik menyeimbangkan ketiganya!

---

## Metodologi: Eksperimen Ilmiah

Eksperimen ini menggunakan desain **studi ablasi (ablation study)** (= metode di mana Anda menghapus atau memvariasikan satu komponen pada satu waktu untuk mengukur dampak individunya — dinamai menurut praktik ilmiah yang secara selektif mengangkat jaringan untuk mempelajari fungsinya):
1. Pilih nilai default: ε=0,2, lr=3e-4, K=10
2. Ubah SATU parameter pada satu waktu
3. Biarkan yang lainnya tetap
4. Bandingkan hasil

Ini memberi tahu kita efek dari SETIAP parameter secara terisolasi.

**Contoh kehidupan nyata:** Menguji apakah pupuk baru membantu tanaman:
- Ganti pupuk, biarkan yang lainnya sama (tanah, air, sinar matahari yang sama)
- Jika tanaman tumbuh lebih baik → pupuk itu membantu!

---

## Temuan Umum dalam Praktik

| Hiperparameter | Terlalu Kecil | Titik Ideal | Terlalu Besar |
|----------------|-----------|------------|-----------|
| **ε (clip)** | Konvergensi lambat | ε ≈ 0,2 | Ketidakstabilan |
| **lr** | Terlalu lambat | 2,5e-4 hingga 3e-4 | Divergensi |
| **K (epoch)** | **Membuang data** (membuang rollout sebelum mengekstrak sinyal penuh) | K = 4-10 | Overfitting pada data rollout usang |
| **n_steps** | Terlalu berisik | 128-2048 | **OOM (Out Of Memory)** (menggunakan terlalu banyak RAM) |
| **batch_size** | Terlalu berisik | 32-256 | **OOM (Out Of Memory)** (menggunakan terlalu banyak RAM) |

"Titik ideal" ini dapat bergeser tergantung pada lingkungannya!

---

## Wawasan Utama: PPO Relatif Tangguh

Dibandingkan dengan algoritma sebelumnya (seperti DQN tanpa target network), PPO relatif tangguh terhadap pilihan hiperparameter. Mekanisme clipping menyediakan jaring pengaman alami.

**Contoh kehidupan nyata:** Mobil dengan rem **ABS** (Anti-lock Braking System — fitur keselamatan yang mencegah roda terkunci saat pengereman keras, menjaga pengemudi tetap terkendali) vs. tanpa ABS:
- Tanpa ABS (DQN): Satu kesalahan belok (hiperparameter buruk) dan Anda tergelincir
- Dengan ABS (PPO): Mobil mengoreksi dirinya sendiri — hiperparameter yang wajar semuanya berfungsi oke

Ketangguhan ini adalah alasan utama PPO adalah algoritma RL yang paling populer dalam praktik!

---

## Poin-Poin Penting

| Konsep | Bahasa Sederhana |
|---------|---------------|
| **Studi ablasi** | Mengubah satu hal pada satu waktu untuk melihat efeknya |
| **Clip epsilon ε** | Batas keamanan — 0,2 biasanya yang terbaik |
| **Learning rate** | **Ukuran langkah** — seberapa banyak bobot jaringan disesuaikan setelah setiap batch (bayangkan sebagai ukuran setiap langkah kaki saat berjalan menuju tujuan). **2,5e-4 hingga 3e-4** adalah notasi ilmiah untuk 0,00025 hingga 0,0003 — ini adalah pengali tak berdimensi, bukan nilai waktu |
| **Update epochs K** | Berapa kali menggunakan kembali setiap batch — 4-10 adalah standar |
| **Random Seeds** | Setiap eksperimen diulangi dengan **seed acak** berbeda (= angka awal yang diberikan ke generator angka acak, yang mengontrol semua pilihan acak dalam pelatihan). Menggunakan beberapa seed mengungkapkan apakah hasilnya konsisten atau hanya karena beruntung |

---

## Ringkasan: Sekilas Metode Policy Gradient

```
REINFORCE              A2C                    PPO
     │                  │                      │
Episode penuh     Pembaruan n-step       N-step + clipping
Sederhana tapi    Lebih cepat tapi       Stabil + efisien
berisik           tidak stabil
Terbaik untuk     Lingkungan tingkat     Lingkungan sulit
lingkungan mudah  menengah               (standar industri)
```

**Jika Anda hanya mempelajari SATU algoritma dari fase ini, pelajarilah PPO.** PPO adalah fondasi dari:
- Pelatihan ChatGPT OpenAI (RLHF menggunakan PPO)
- Lanjutan AlphaGo DeepMind
- Sebagian besar riset robotika modern
- AI pemain video game
