# Conservative Q-Learning (CQL) 🛡️

## Apa Itu?

Bayangkan Anda belajar berinvestasi uang dengan membaca buku besar raksasa berisi
transaksi saham masa lalu yang dilakukan oleh orang lain. Buku besar itu berisi beli, jual, dan tahan —
tetapi **tidak ada catatan transaksi yang tidak pernah dilakukan siapa pun**.

Sekarang bayangkan seorang siswa yang terlalu percaya diri melihat buku besar itu dan berkata:
*"Bagaimana jika seseorang membeli tiket lotre setiap hari Senin? Itu pasti akan menjadi transaksi yang luar biasa!"*

Masalahnya: **buku besar tidak memiliki data tentang pembelian lotre hari Senin**, jadi
siswa tersebut hanya berhalusinasi. Namun transaksi hasil halusinasi itu terlihat bagus di atas
kertas, sehingga "kebijakan" siswa tersebut terus ingin melakukannya.

Masalah halusinasi itu adalah **pergeseran distribusi (distribution shift)**: pembelajar offline
menyukai tindakan yang tidak pernah diuji oleh dataset, karena tidak ada data yang membantah
optimisme tersebut. CQL adalah penawarnya.

---

## Bagaimana Q-Learning Menjadi Salah Secara Offline

Target Q-learning normal adalah:

```
target(s, a) = r + γ · max_{a'} Q(s', a')
```

`max_{a'}` itulah bahayanya. Ketika dataset tidak pernah mencatat tindakan `a'`
dalam status `s'`, jaringan hanya *menebak* nilai-Q — dan jaringan saraf cenderung
**melebih-lebihkan (over-estimate)** Q untuk input yang tidak terlihat. Target tersebut mewarisi
perkiraan yang berlebihan, jaringan belajar untuk memprediksi angka yang lebih besar itu, dan pada
langkah berikutnya kita melakukan **ekstrapolasi** (memproyeksikan bahkan lebih jauh melampaui apa pun yang
didukung data) bahkan lebih tinggi lagi. Kebijakan tersebut mengejar fatamorgana.

Jika Anda bisa terus mengumpulkan lebih banyak data, ini akan memperbaiki diri sendiri (tindakan
fatamorgana itu ternyata buruk dalam kenyataan). Tetapi **dalam RL offline Anda tidak bisa mengumpulkan lebih banyak data.** Fatamorgana itu abadi.

---

## Trik CQL

CQL (Kumar et al., 2020) menambahkan istilah penalti ke loss:

```
cql_loss(s)  =  log Σ_a exp Q(s, a)   -   Q(s, a_dataset)
```

Dua bagian:

1. **`log Σ_a exp Q(s, a)`** (baca: *"log-sum-exp di semua tindakan"*) adalah sebuah
   **soft maximum** di semua tindakan — pendekatan yang halus dan dapat dideferensiasi dari `max` yang mempertimbangkan setiap tindakan sekaligus daripada memilih satu pemenang secara kaku. Memberikan penalti padanya akan mengecilkan nilai-Q
   **secara menyeluruh** (mendorong semua prediksi
   turun secara seragam) — terutama untuk tindakan dengan Q *tertinggi*, yang merupakan tempat
   halusinasi berada.
2. **`- Q(s, a_dataset)`** memberi imbalan Q yang tinggi pada tindakan yang sebenarnya
   dicatat oleh dataset — melindungi nilai-Q yang masuk dalam distribusi dari penyusutan di atas.

Efek bersih: **Q ditarik turun pada tindakan yang tidak terlihat, ditarik naik pada tindakan yang
terlihat.** Q yang dipelajari menjadi *batas bawah (lower bound)* dari Q yang sebenarnya. Kebijakan
**`argmax`** (aturan yang cukup memilih tindakan dengan Q tertinggi) berhenti mengejar fatamorgana.

Loss penuh:

```
L  =  Bellman_MSE   +   α · cql_loss
```

(Di mana **`Bellman_MSE`** adalah kesalahan standar dari Q-learning normal,
mengukur seberapa besar ketidaksesuaian tebakan jaringan saat ini dengan tebakan masa depannya sendiri).

`α` adalah tombol konservatisme. Terlalu kecil → pergeseran distribusi merayap kembali.
Terlalu besar → agen sangat konservatif sehingga tidak pernah berkembang melampaui data.

---

## Contoh Kehidupan Nyata

- **Pelatih catur konservatif.** Anda hanya bisa belajar dari permainan yang sudah
  dimainkan. Pelatih yang ceroboh berkata "langkah hipotesis tanpa preseden ini
  bisa jadi brilian!" CQL adalah pelatih yang berkata "kita tidak punya data tentang
  itu — mari kita tetap berpegang pada langkah yang sudah dicoba oleh pemain asli."
- **Pilihan menu restoran.** Ulasan Yelp tidak pernah mencakup item di luar menu. Kebijakan naif akan merekomendasikan item di luar menu berdasarkan rating bintang lima hasil halusinasi. CQL hanya merekomendasikan apa yang sudah cukup sering dipesan untuk dipercaya.
- **Robot yang memegang benda dari log.** Robot memiliki video memegang cangkir, botol, dan buku — tetapi tidak pernah pisau. CQL menolak untuk merekomendasikan dengan percaya diri "pegang pisau pada bilahnya."

---

## Apa yang Dilakukan Kode Kami

Skrip `cql.py`:

1. **Memuat keempat dataset** yang dibuat oleh `d4rl_dataset.py`.
2. **Memilih `medium-replay`** sebagai set pelatihan — ini yang paling realistis
   (kualitas campuran) dan paling merusak bagi metode naif.
3. **Melatih tiga agen secara murni offline**, dalam kondisi identik kecuali
   untuk `α`:
   - `α = 0`   →  DQN offline naif (tanpa penalti — baseline yang rusak)
   - `α = 1.0` →  CQL ringan
   - `α = 5.0` →  CQL kuat
4. **Mengevaluasi masing-masing setiap 2.500 langkah gradien** dengan menjalankannya secara
   rakus di lingkungan yang sebenarnya (10 episode). Ini adalah *satu-satunya* kontak dengan lingkungan;
   pelatihan itu sendiri tidak pernah melihat lingkungan.
5. **Membuat plot kurva pembelajaran** ke `outputs/cql.png`.

---

## Apa yang Seharusnya Anda Lihat

Jalannya program biasanya mencetak sesuatu seperti:

```
Return evaluasi akhir (rata-rata di 10 episode, greedy):
  naive offline DQN (alpha=0)         ->  ~30-150  (tidak stabil; sering jatuh)
  CQL (alpha=1.0)                     ->  ~300-450
  CQL (alpha=5.0)                     ->  ~450-500
```

Dalam plot kurva pembelajaran:

- **Kurva merah** (`α = 0`) naik di awal lalu sering **jatuh dari tebing**
  setelah halusinasi pergeseran distribusi menginfeksi **target Bellman**
  (angka yang kita gunakan sebagai "jawaban yang benar" saat melatih jaringan-Q:
  `r + γ · max Q(s', ·)`). Saat nilai-Q fatamorgana mencemari target tersebut,
  setiap langkah gradien membuat keadaan menjadi lebih buruk. **Bellman loss** (MSE
  antara prediksi jaringan-Q dan target Bellman) terlihat baik-baik saja — itulah **pengkhianatan** dari masalah ini: jaringan sangat konsisten dengan keyakinannya yang salah, sehingga loss tidak memberikan peringatan.
- **Kurva oranye** (`α = 1.0`) naik lebih lambat tetapi **tetap bertahan**.
- **Kurva hijau** (`α = 5.0`) adalah yang paling stabil dan biasanya yang terbaik.

Panel Bellman-loss menunjukkan tanda lain: loss DQN naif dapat tetap kecil sementara
kebijakannya mengerikan, karena jaringan konsisten secara internal dengan halusinasinya sendiri.

---

## Di Mana Posisi CQL dalam Bidang Ini

CQL adalah penemuan besar karena memberikan perbaikan yang berprinsip dan sederhana pada
pergeseran distribusi. Silsilahnya:

```
DQN (online)
   │
   ▼
Naive offline DQN  ── rusak karena pergeseran distribusi
   │
   ▼
CQL (Kumar 2020)    ── tambahkan penalti konservatif: Q adalah batas bawah
   │
   ▼
IQL (Kostrikov 2021)  ── tidak pernah menanyakan Q pada tindakan yang tidak terlihat
   │
   ▼
Decision Transformer (Chen 2021)  ── lewati Q sepenuhnya, perlakukan RL sebagai pemodelan urutan
                                      (prediksi *tindakan berikutnya* dengan status masa lalu dan
                                       return total yang diinginkan, persis seperti LLM
                                       memprediksi kata berikutnya)
```

Setiap langkah dalam silsilah ini adalah jawaban berbeda untuk pertanyaan yang sama:
**bagaimana cara menghindari menanyakan jaringan-Q saya tentang hal-hal yang belum pernah ia lihat?**

---

## Kata Kunci untuk Diingat

| Kata | Makna |
|------|---------|
| **Distribution shift** | Kebijakan yang dilatih menginginkan tindakan di luar data |
| **Out-of-distribution (OOD)** | Pasangan (s, a) yang tidak pernah dicatat oleh dataset |
| **Q Sebenarnya (True Q)** | Return masa depan yang diharapkan secara nyata untuk mengambil tindakan `a` di status `s`, jika kita bisa mengukurnya dengan sempurna |
| **Q Konservatif** | Fungsi-Q yang dipelajari yang mencoba tetap berada di bawah Q sebenarnya daripada menjanjikan secara berlebihan |
| **Logsumexp** | Pendekatan halus dan dapat dideferensiasi dari `max` |
| **Alpha (α)** | Tombol konservatisme CQL — seberapa keras menekan Q pada tindakan OOD |

---

## Ringkasan Satu Kalimat

> **CQL menambahkan "penalti pesimisme" yang menghukum nilai-Q tinggi pada tindakan yang tidak pernah dicoba oleh dataset — sehingga kebijakan tidak bisa jatuh cinta pada halusinasi.**
