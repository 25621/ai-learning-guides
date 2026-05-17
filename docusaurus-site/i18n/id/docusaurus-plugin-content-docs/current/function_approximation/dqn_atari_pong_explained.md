# DQN pada Atari Pong 🏓

## Apa itu Atari Pong?

Pong adalah salah satu video game tertua yang pernah dibuat — ini seperti tenis meja digital! Dua pedal memantulkan bola bolak-balik. Anda memenangkan poin jika lawan meleset dari bola. Permainan berakhir saat seseorang mencapai 21 poin.

Dalam versi kami, AI mengendalikan satu pedal. Lawan (komputer) mengendalikan yang lain. Permainan selalu dimulai pada skor −21 (kemungkinan terburuk). Agen yang baik mencapai 0 atau +21.

---

## Mengapa Pong Sulit bagi AI?

Dalam CartPole, robot dapat MELIHAT angka secara langsung (sudut tiang, kecepatan kereta...). Dalam Pong, yang dilihatnya hanyalah **piksel mentah** — ribuan titik berwarna kecil di layar!

```
Input CartPole: [0.02, −0.14, 0.01, −0.23]   ← 4 angka, mudah!
Input Pong:     [kisi piksel: 210×160×3]      ← 100.800 angka, JAUH lebih sulit!
```

Robot harus mencari tahu dari piksel:
- Di mana pedal saya?
- Di mana bolanya?
- Apakah bola bergerak ke kiri atau ke kanan?
- Seberapa cepat?

Manusia melakukan ini secara otomatis (kita memiliki penglihatan yang luar biasa!). Bagi AI, ini adalah tantangan besar.

---

## Melihat Gerakan: Penumpukan Bingkai (Frame Stacking) 🎬

Satu bingkai (tangkapan layar) tidak memberi tahu Anda apakah bola bergerak ke kiri atau ke kanan. Anda perlu melihat BEBERAPA bingkai untuk memahami gerakan — persis seperti bagaimana buku flip hanya berfungsi saat Anda membalik banyak halaman.

**Penumpukan Bingkai:** Masukkan 4 bingkai terakhir ke dalam jaringan secara bersamaan.

```
Bingkai 1: bola di posisi 40
Bingkai 2: bola di posisi 43    → Tumpuk 4 bingkai ini → Jaringan melihat GERAKAN!
Bingkai 3: bola di posisi 46
Bingkai 4: bola di posisi 49
```

Jaringan sekarang dapat menyimpulkan: "bola bergerak ke kanan dengan kecepatan 3"

**Contoh kehidupan nyata:** Menonton film vs melihat satu bingkai. Satu bingkai dari balapan mobil hanyalah gambar buram. Tonton 4 bingkai, dan Anda dapat mengetahui mobil mana yang lebih cepat!

---

## Melihat dengan CNN 🔍

Untuk input piksel, kita menggunakan jaringan saraf khusus yang disebut **Convolutional Neural Network (CNN)**. Alih-alih melihat semua piksel sekaligus, CNN menggunakan jendela geser untuk mendeteksi pola — seperti mata yang memindai gambar.

```
Piksel mentah (84×84×4 bingkai)
       ↓
Lapisan Konv 1 (filter 8×8, stride 4) → menemukan tepi dan bentuk
       ↓
Lapisan Konv 2 (filter 4×4, stride 2) → menemukan objek (pedal, bola)
       ↓
Lapisan Konv 3 (filter 3×3, stride 1) → menemukan hubungan
       ↓
Flatten → 512 neuron → nilai-Q (satu per tindakan)
```

**Contoh kehidupan nyata:** Saat Anda mencari teman di tengah keramaian, otak Anda pertama-tama memperhatikan bentuk (seorang manusia), lalu fitur (warna rambut), lalu detail (wajah mereka). CNN bekerja dengan cara yang sama — dari pola sederhana ke pola yang kompleks!

---

## Pemrosesan Awal: Mengecilkan Dunia

Bingkai Pong berukuran 210×160 piksel berwarna. Itu terlalu besar! Kami memproses setiap bingkai:

1. **Grayscale** — warna tidak masalah untuk Pong (bolanya selalu putih)
2. **Ubah ukuran ke 84×84** — lebih kecil = pelatihan lebih cepat, tetapi masih cukup jelas untuk dilihat
3. **Normalisasi ke [0,1]** — bagi nilai piksel dengan 255 sehingga menjadi angka kecil

**Contoh kehidupan nyata:** Seperti membuat fotokopi dengan ukuran 50%. Detail penting (bola, pedal) masih terlihat, hanya saja lebih kecil. Fotokopi juga tidak peduli dengan warna!

---

## Kliping Imbalan: Memperlakukan Semua Game Secara Setara ✂️

Di Pong, Anda mendapatkan +1 jika mencetak gol, −1 jika kebobolan. Di beberapa game Atari lainnya, skor bisa mencapai ribuan!

Kami melakukan **kliping imbalan (reward clipping)** ke [−1, +1] sehingga jaringan tidak peduli dengan skala imbalan. Kode yang sama ini dapat melatih game Atari APA PUN tanpa menyetel skala imbalan.

---

## Berapa Lama Waktu Pelatihan?

| Durasi Pelatihan | Apa yang Dipelajari Agen |
|---|---|
| 100 ribu langkah | Sebagian besar acak, nyaris tidak bereaksi |
| 1 juta langkah | Mulai bergerak menuju bola kadang-kadang |
| 5 juta langkah | Mengembalikan beberapa pukulan |
| 10 juta langkah | Bermain kompetitif, mungkin menang beberapa kali |
| 20 juta+ langkah | Sering mengalahkan lawan komputer |

Demo kami berjalan selama **300 ribu langkah** — cukup untuk melihat arsitektur pelatihan berfungsi dan mengamati pembelajaran awal, tetapi tidak cukup untuk menguasai permainan.

**Contoh kehidupan nyata:** Belajar piano butuh waktu berbulan-bulan. Sesi latihan 10 menit menunjukkan Anda melakukannya dengan benar, tetapi jangan berharap untuk segera melakukan konser!

---

## Apa yang Ditemukan Kode Kami

Setelah 300 ribu langkah pada Pong:
- Agen memulai dengan skor sekitar −20 (nyaris tidak mengembalikan bola)
- Di akhir, biasanya meningkat menjadi sekitar −15 hingga −10
- Kurva pembelajaran menunjukkan peningkatan bertahap dari permainan acak

Untuk melihat performa Pong yang benar-benar kompetitif, Anda perlu menjalankan sekitar 10 juta+ langkah dengan GPU. Implementasinya lengkap dan benar — hanya butuh lebih banyak waktu!

---

## Kosakata Kunci

| Kata | Makna |
|------|---------|
| **CNN** | Convolutional Neural Network — dikhususkan untuk input gambar |
| **Frame Stacking** | Memasukkan beberapa bingkai berturut-turut untuk menangkap gerakan |
| **Preprocessing** | Mentransformasi bingkai mentah (grayscale, ubah ukuran, normalisasi) sebelum dimasukkan ke jaringan |
| **Reward Clipping** | Membatasi imbalan ke [−1, +1] agar berfungsi di berbagai game berbeda |
| **ALE** | Arcade Learning Environment — pustaka yang menjalankan game Atari |

---

## Pencapaian Bersejarah

Saat DeepMind menerbitkan DQN pada tahun 2015, dunia takjub. SATU algoritma tunggal, dengan arsitektur yang SAMA, belajar memainkan 49 game Atari yang berbeda — banyak di antaranya pada level manusia super — hanya dari piksel mentah dan skor!

Sebelum DQN, orang mengira Anda perlu memprogram strategi secara manual untuk setiap game. DQN menunjukkan bahwa pembelajar tujuan umum dapat mencari tahu semuanya sendiri. Itu adalah momen bersejarah dalam AI!
