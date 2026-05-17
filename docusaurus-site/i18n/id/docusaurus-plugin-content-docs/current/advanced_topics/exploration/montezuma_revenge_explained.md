# Melatih pada Montezuma's Revenge 🏛️🔑

## Mengapa Game Ini Terkenal (dalam lingkaran RL) {#why-this-game-is-famous-in-rl-circles}

Pada tahun 2015, DQN milik DeepMind belajar memainkan puluhan game Atari pada level manusia super dari piksel mentah. Hal ini menjadi berita utama. Namun tersembunyi di dalam tabel hasil ada satu game di mana DQN mencetak skor **0** — sama dengan tidak melakukan apa-apa sama sekali: **Montezuma's Revenge**.

Mengapa? Lihat apa yang diminta game ini dari Anda di ruangan pertama:

1. Turun tangga.
2. Berjalan di atas langkan.
3. Melompati tengkorak yang menggelinding (salah waktu → Anda mati).
4. Naik tangga lain.
5. Ambil kunci.

Itu kira-kira **100 tekanan tombol yang presisi**, dan game tersebut tidak memberi Anda **satu poin pun** sampai kunci ada di tangan. Sinyal imbalannya adalah **nol** datar tanpa fitur untuk seluruh urutan pembukaan.

Agen RL normal belajar dengan menyesuaikan diri terhadap imbalan yang benar-benar diterimanya. Jika imbalannya nol di mana-mana yang dapat dijangkaunya, maka *tidak ada yang bisa dipelajari* — itu seperti mencoba menemukan dasar lembah yang benar-benar datar dengan meraba arah menurun. Jadi DQN hanya berkedut di sekitar platform awal selamanya. Montezuma menjadi standar (benchmark) untuk **eksplorasi sulit**: game yang hanya bisa Anda kalahkan jika Anda bereksplorasi secara *cerdas*, bukan secara acak.

Terobosan datang pada tahun 2018 dengan **Random Network Distillation (RND)** — dan triknya persis seperti subjek item pekerjaan 1: tambahkan **bonus keingintahuan intrinsik** sehingga agen memberi imbalan *pada dirinya sendiri* karena mencapai layar baru, dan tiba-tiba ia memiliki sinyal padat yang menariknya lebih dalam ke level tersebut. RND mendapatkan skor manusia super di Montezuma. (Kemudian: Go-Explore, Agent57, …)

## Contoh Imbalan Jarang "gaya Montezuma" di Kehidupan Nyata {#real-life-examples-of-montezuma-style-sparse-reward}

- **Kunci kombinasi / perburuan harta karun dengan petunjuk samar.** Tidak ada nilai parsial. Anda berada di angka nol sampai tiba-tiba Anda mendapatkan hadiahnya.
- **Mendapatkan makalah yang diterima, atau startup yang menguntungkan.** Berbulan-bulan tanpa imbalan eksternal, lalu (mungkin) satu yang besar.
- **Rute speedrun video game.** Puluhan input bingkai-sempurna (frame-perfect) berturut-turut tanpa umpan balik sampai triknya berhasil atau tidak.
- **Escape rooms.** Ruangan tersebut hampir tidak memberi tahu Anda apa pun sampai Anda merangkai beberapa penemuan bersama-sama.

Dalam semua ini, "coba saja barang acak" tidak ada harapan. Anda perlu bereksplorasi secara *sistematis* — dan sinyal internal "ooh, itu baru, lanjut terus" adalah apa yang membuat Anda tetap sistematis.

## Mengapa Kami Tidak Benar-benar Melatih pada Pixel Montezuma di Sini {#why-we-dont-actually-train-on-pixel-montezuma-here}

Melakukan hal yang *nyata* dengan benar berarti:

- jaringan konvolusional untuk melihat layar RGB 210×160,
- penumpukan bingkai (sehingga agen dapat mengetahui ke arah mana tengkorak bergerak),
- modul RND (dua jaringan lagi: "target" acak tetap dan "prediktor" terlatih),
- dan **puluhan juta bingkai lingkungan** — banyak jam-GPU.

Itu adalah proyek penelitian, bukan skrip pengajaran. Jadi `montezuma_revenge.py` melakukan dua hal jujur sebagai gantinya:

### 1. Ia *menyentuh* game aslinya (jika `ale-py` terinstal) {#1-it-touches-the-real-game-if-ale-py-is-installed}

Ia memuat `ALE/MontezumaRevenge-v5` melalui Gymnasium, menjalankan **agen acak seragam selama 2000 langkah**, dan melaporkan total imbalan game. Angka yang dicetaknya hampir selalu **0.0** — frasa abstrak "imbalan jarang" berubah menjadi fakta konkret yang dijalankan sendiri. Jika paket Atari tidak terinstal, ia mencetak perintah `pip install` satu baris dan melanjutkan.

### 2. Ia melatih agen tabular pada *model skala*: `MiniMontezumaEnv` {#2-it-trains-a-tabular-agent-on-a-scale-model-minimontezumaenv}

Ini adalah gridworld kecil dengan *kerangka* yang sama dengan ruangan pertama Montezuma:

```
###############
#S....#.......#
#.....#.......#
#.....#...T...#     S = mulai
#.....D.......#     K = kunci      D = pintu (hanya bisa dilewati saat membawa kunci)
#..K..#.......#     T = harta karun (SATU-SATUNYA petak yang memberi imbalan)
###############
```

Untuk menang Anda harus: berjalan ke **kunci** (~6 gerakan), ambil; berjalan ke **pintu** (~4 gerakan) — yang sekarang terbuka; berjalan menembus dan mencapai **harta karun** (~5 gerakan). Sekitar **15 gerakan sempurna**, dengan **nol umpan balik sampai harta karun**. Flag `has_key` adalah bagian dari status agen, jadi begitu Anda mengambil kunci, ada seluruh ruangan kedua dari status *baru* untuk ditemukan — persis seperti layar baru yang terbuka di game aslinya.

Kami kemudian melatih sebuah **tabular Q-learner** biasa dua kali:

| Agen | Hasil pada MiniMontezuma |
|-------|--------------------------|
| **tanpa keingintahuan (epsilon-greedy)** | Return tetap di **0** untuk seluruh 1.500 episode. Ia bahkan tidak pernah mencapai kunci. (Terdengar akrab? Itulah DQN pada game aslinya.) |
| **dengan bonus keingintahuan kesalahan-prediksi** | Mencapai harta karun dalam ~20–25 episode dan kemudian mempelajari **rute 15-langkah optimal**. (Itu adalah ide RND, diperkecil agar pas dalam tabel-Q.) |

Gambar menunjukkan dua kurva pembelajaran berdampingan, ditambah rute aktual yang dipelajari agen yang penasaran, digambar pada kisi (mulai → kunci → pintu → harta karun). Skrip juga mencetak rute tersebut sebagai bingkai ASCII.

## Pelajaran {#the-lesson}

> **"Imbalan jarang" bukan sekadar keunikan dari satu game Atari yang aneh — ini adalah default di dunia mana pun di mana kesuksesan membutuhkan urutan tindakan yang panjang dan spesifik.** Agen imbalan-saja (vanilla DQN) secara harfiah tidak dapat memulai: tidak ada gradien untuk diikuti. Bonus keingintahuan menciptakan satu gradien — sinyal padat yang dihasilkan sendiri "ini baru, lanjut terus" — dan sinyal itulah yang membawa agen melintasi gurun angka nol menuju imbalan nyata pertama. Segala sesuatu setelah itu (RND, Go-Explore, Agent57) adalah versi jaringan saraf yang diperbesar dari langkah yang sama.

## Kata Kunci untuk Diingat {#key-words-to-remember}

| Kata | Makna |
|------|---------|
| **Hard exploration** | Masalah di mana Anda hanya berhasil dengan bereksplorasi secara cerdas; eksplorasi acak gagal |
| **Sparse reward** | Imbalan bernilai nol hampir di mana-mana; Anda mendapatkannya hanya setelah urutan benar yang panjang |
| **Montezuma's Revenge** | Game Atari di mana agen deep-RL klasik (DQN, A3C) mencetak skor 0 — benchmark kanonik untuk eksplorasi sulit |
| **RND (Random Network Distillation)** | Metode tahun 2018 yang mengalahkan Montezuma menggunakan bonus keingintahuan kesalahan-prediksi |
| **Go-Explore** | "Ingat status yang menjanjikan, kembali ke sana, lalu bereksplorasi dari sana" — pemecah Montezuma lainnya |
| **Model skala (Scale model)** | Lingkungan kecil dan murah yang mempertahankan *struktur* masalah sulit sehingga Anda dapat mempelajarinya dengan cepat |

## Ringkasan Satu Kalimat {#one-sentence-summary}

> **Montezuma's Revenge adalah game yang mengajarkan RL bahwa "imbalan yang tidak pernah Anda terima tidak dapat mengajarkan apa pun kepada Anda" — dan solusinya, dulu dan sekarang, adalah bonus keingintahuan yang membiarkan agen menghadiahi dirinya sendiri karena bereksplorasi sampai ia menemukan hadiah yang sebenarnya.**
