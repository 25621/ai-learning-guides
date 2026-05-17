# Dataset Benchmark D4RL 📦

## Apa Itu?

Bayangkan Anda ingin mengajari robot cara membalikkan panekuk. Membiarkannya berlatih
di atas kompor sungguhan selama sebulan akan lambat, berbahaya, dan mahal. Tetapi Anda
memiliki rekaman video selama sepuluh tahun tentang koki yang membalikkan panekuk
(ada yang bagus, ada yang buruk, ada yang acak). Bisakah Anda mengajari robot *hanya dari data itu*, tanpa membiarkannya menyentuh wajan sungguhan?

Itulah **pembelajaran penguatan offline (offline reinforcement learning)**. Agen belajar dari dataset pengalaman masa lalu yang tetap — tidak ada lingkungan langsung. Bagian tersulitnya adalah agen tidak pernah bisa *mencoba* apa yang dipelajarinya sampai akhir.

Agar adil untuk dipelajari, komunitas riset membutuhkan *dataset standar*. Itulah **D4RL** (**D**atasets for **D**eep **D**ata-**D**riven **R**einforcement **L**earning): kumpulan transisi yang direkam sebelumnya untuk tugas kontrol klasik, yang dirilis oleh UC Berkeley pada tahun 2020. Setiap makalah melatih pada byte yang sama, sehingga hasilnya dapat dibandingkan.

---

## Apa Saja yang Ada di Dataset D4RL?

Untuk setiap tugas, D4RL menyediakan **empat tingkat kualitas**:

| Tingkat | Dari mana data berasal | Mengapa ini penting |
|-------|---------------------------|----------------|
| **random**        | Kebijakan yang memilih tindakan secara acak seragam | Kasus terburuk: bisakah Anda tetap mempelajari sesuatu yang berguna? |
| **medium**        | Kebijakan yang dilatih sebagian (sekitar setengah dari skor pakar) | Realistis: sebagian besar data yang dicatat adalah kualitas menengah |
| **expert**        | Kebijakan yang hampir konvergen | Kasus terbaik: bisakah Anda menyamai kebijakan sumber? |
| **medium-replay** | *Seluruh replay buffer* yang digunakan untuk melatih kebijakan medium | Campuran: berisi kegagalan awal DAN keberhasilan kemudian |

Perbedaan antara `medium` dan `medium-replay` sangat penting:
- **`medium`** dihasilkan dengan mengambil satu kebijakan "rata-rata" yang tetap dan membiarkannya memainkan banyak game. Semua data mencerminkan tingkat keterampilan rata-rata yang stabil ini.
- **`medium-replay`** adalah log historis. Ia berisi semua pengalaman yang dikumpulkan *saat belajar* dari awal hingga tingkat medium. Ia mencampur transisi **buruk dan baik** — persis seperti tampilan log di dunia nyata (upaya pertama robot yang canggung *dan* perilaku selanjutnya yang halus, semuanya dalam satu wadah).

---

## Contoh Dataset Offline di Kehidupan Nyata

- **Rekam medis.** Bertahun-tahun pasangan (status_pasien, pengobatan, hasil).
  Anda tidak bisa mengacak pengobatan pada orang yang masih hidup, tetapi Anda bisa belajar
  kebijakan yang lebih baik dari log tersebut.
- **Log obrolan layanan pelanggan.** Jutaan catatan (pesan_pengguna, balasan_agen, kepuasan). Latih asisten yang lebih baik tanpa mengganggu lebih banyak pengguna.
- **Data armada mengemudi otonom.** Setiap mobil Tesla / Waymo mengunggah hasil
  perjalanannya. Armada tersebut adalah dataset medium-replay raksasa.
- **Sistem rekomendasi.** Log klik dari tahun lalu adalah dataset yang dibekukan:
  Anda tidak bisa menampilkan kembali iklan yang sama kepada pengguna yang sama.

Dalam keempat kasus tersebut, **Anda tidak bisa meminta sampel baru dari lingkungan.** Dataset adalah apa yang Anda miliki. Selamanya.

---

## Apa yang Dilakukan Kode Kami

Dataset D4RL yang asli direkam pada tugas lokomosi MuJoCo (Multi-Joint dynamics with Contact) (seperti HalfCheetah, Hopper, Walker2d, Ant — ini adalah simulasi fisika 3D canggih di mana robot virtual belajar berjalan dan berlari). MuJoCo berat untuk diinstal, jadi kami membuat ulang **struktur empat tingkat yang sama pada CartPole-v1** — lingkungan pemula standar dari fase-fase sebelumnya. Pelajarannya dapat ditransfer secara langsung.

Skrip `d4rl_dataset.py`:

1. **Melatih sebuah DQN** (Deep Q-Network, algoritma RL standar) pada CartPole sampai ia menyelesaikan tugas (return ≥ 475).
2. **Mengambil dua checkpoint** di sepanjang jalan:
   - "medium" — saat return terbaru melewati 150
   - "expert" — saat return terbaru melewati 475
3. **Mengambil seluruh replay buffer kebijakan medium** — setiap transisi yang pernah dilihatnya. Itulah dataset "medium-replay" kami.
4. **Menjalankan tiga kebijakan baru** masing-masing untuk 10.000 transisi:
   - `random`   — acak seragam
   - `medium`   — checkpoint medium + noise ε=0.10
   - `expert`   — checkpoint pakar + noise ε=0.02
5. **Menyimpan empat file `.npz`** (format array terkompresi NumPy) di `outputs/`, masing-masing dengan array `obs / action / reward / next_obs / terminal`.

Keempat file ini adalah input untuk `cql.py` dan `behavioral_cloning.py`.

---

## Apa yang Seharusnya Anda Lihat Saat Menjalankannya

Ringkasan teks biasa dicetak ke konsol dan disimpan ke `outputs/d4rl_summary.txt`:

```
dataset         |   N    |  return rata-rata |  min  |  max
------------------------------------------------------------
random          | 10000  |          ~22      |    ~9 |   ~80
medium          | 10000  |         ~180      |   ~50 |  ~500
expert          | 10000  |         ~490      |  ~400 |   500
medium-replay   | 10000  |          ~60      |    ~9 |  ~200
```

Ini juga menghasilkan histogram (`outputs/d4rl_returns.png`) yang menunjukkan bagaimana keempat dataset tersebut tumpang tindih. Fitur utama yang perlu diperhatikan:

- **Random** berkumpul di sekitar 20 (panjang rata-rata episode CartPole acak).
- **Expert** berkumpul di batas atas 500.
- **Medium** berada di antaranya, dengan variansi tinggi.
- **Medium-replay** memiliki "ekor" kanan yang panjang — sebagian besar terdiri dari percobaan awal yang gagal (return rendah) tetapi memiliki ekor yang memanjang ke return yang lebih tinggi saat agen belajar.

---

## Mengapa Dataset Itu Penting

Dataset mana pun yang Anda gunakan untuk melatih algoritma offline Anda, Anda memberikan *batas atas (ceiling)* pada apa yang mungkin:

- **Dari `expert`** — bahkan algoritma bodoh seperti BC (Behavioral Cloning, yang hanya menyalin data persis) dapat bekerja dengan baik, karena semua datanya bagus.
- **Dari `random`** — Anda membutuhkan algoritma pintar yang dapat *merangkai (stitch together)* transisi bagus yang langka (menemukan jalan menuju sukses dengan menggabungkan urutan pendek tindakan bagus dari berbagai percobaan). BC akan gagal total.
- **Dari `medium-replay`** — yang paling realistis dan paling menarik. Algoritma yang baik (seperti **CQL** — Conservative Q-Learning, yang menghindari rasa percaya diri berlebihan tentang tindakan yang belum pernah dilihatnya) terkadang dapat **mengalahkan kualitas rata-rata data** karena mereka mengekstrak struktur dari sinyal campuran. Algoritma bodoh (BC) akan kembali ke rata-rata (regress to the mean).

Kita akan melihat cerita ini persis di dua skrip berikutnya.

---

## Kata Kunci untuk Diingat

| Kata | Makna |
|------|---------|
| **Offline RL**         | Melatih dari dataset tetap; tidak ada interaksi lingkungan yang diperbolehkan |
| **Behaviour policy**   | Kebijakan yang *menghasilkan* dataset |
| **Dataset quality**    | Seberapa baik kebijakan perilaku tersebut (random / medium / expert) |
| **Replay buffer**      | Sejarah lengkap transisi yang dilihat selama sesi pelatihan |
| **Distribution shift** | Kesenjangan antara tindakan dalam dataset dan tindakan yang ingin diambil oleh kebijakan yang Anda latih. Karena dataset tidak pernah menunjukkan apa yang terjadi saat kebijakan baru mencoba sesuatu yang tidak tercatat, estimasi nilai algoritma untuk tindakan baru tersebut bisa menjadi sangat salah. |

---

## Ringkasan Satu Kalimat

> **D4RL membekukan RL menjadi benchmark gaya pembelajaran terawasi (supervised learning): byte yang sama untuk semua orang, tidak ada kecurangan lingkungan, semoga algoritma terbaik yang menang.**
