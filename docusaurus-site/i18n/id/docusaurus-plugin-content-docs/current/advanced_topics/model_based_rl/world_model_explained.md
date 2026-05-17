# Melatih World Model: Mengajari Agen untuk Bermimpi 🌍

## Apa Itu "World Model"?

Sebuah **world model** adalah *salinan internal alam semesta* milik agen. Berikan ia sebuah status dan sebuah tindakan, dan ia memprediksi apa yang akan terjadi selanjutnya:

```
(status, tindakan) ──► Jaringan Saraf ──► (status_berikutnya, imbalan)
```

Ini bukan dunia nyata — ini adalah **simulator yang dibangun agen untuk dirinya sendiri** dengan melihat kenyataan dan belajar untuk menirunya.

Setelah dilatih, model tersebut membiarkan agen mengajukan pertanyaan "bagaimana-jika" tanpa melakukan tindakan nyata apa pun:

> *"Jika saya mendorong ke kiri sekarang dan kemudian ke kanan dua kali, di mana saya akan berakhir? Apakah tiangnya akan jatuh?"*

Agen dapat memikirkan seratus rencana di dalam modelnya dalam waktu yang sama dengan waktu yang dibutuhkan untuk melakukan satu gerakan nyata. Itulah intinya.

---

## Analogi Kehidupan Nyata

Pikirkan tentang bagaimana *Anda* menyelesaikan teka-teki. Anda tidak secara fisik memindahkan setiap kepingan ke setiap slot. Anda **membayangkan** apa yang terjadi jika kepingan A diletakkan di sini. Jika simulasi mental itu terlihat salah, Anda menolaknya bahkan sebelum mengangkat jari.

Otak Anda memiliki world model terpelajar — dibangun dari bertahun-tahun melihat bagaimana objek berperilaku — yang membiarkan Anda mensimulasikan hasil sebelum berkomitmen.

Contoh lainnya:

- **Pemain catur** membayangkan gerakan beberapa giliran ke depan.
- **Pengemudi** berpikir, "Jika saya mengerem sekarang, mobil di belakang memiliki cukup ruang."
- **Seorang anak** menyusun balok: "Jika saya menaruh yang besar di atas, menaranya akan goyah." (Mereka mempelajari model ini dengan menjatuhkan menara sebelumnya.)

Dalam setiap kasus, **model mental + imajinasi = keputusan yang lebih baik dengan risiko yang lebih sedikit**.

---

## Bagaimana Agen Membangun Modelnya?

Ia hanya **menonton**. Secara khusus:

1. **Kumpulkan data.** Biarkan kebijakan apa pun (bahkan acak) berinteraksi dengan lingkungan nyata untuk sementara waktu. Simpan setiap transisi:
   ```
   (status, tindakan, imbalan, status_berikutnya)
   ```
2. **Latih jaringan saraf** untuk memprediksi `status_berikutnya` dan `imbalan` dari `(status, tindakan)`. Ini adalah pembelajaran terawasi (supervised learning): setiap transisi yang disimpan adalah contoh berlabel di mana inputnya adalah "apa yang dilihat dan dilakukan agen" dan labelnya adalah "apa yang sebenarnya terjadi selanjutnya."
3. **Validasi.** Sisihkan 10% data dan periksa prediksi model terhadap data nyata. Error yang rendah berarti model telah menangkap **dinamika** lingkungan: bagaimana status berubah setelah tindakan dilakukan.

Trik yang kami gunakan: alih-alih memprediksi `status_berikutnya` secara langsung, prediksi **delta** `status_berikutnya − status`. Sebagian besar fisika bersifat bertahap ("kereta bergerak sedikit saja"), dan target yang kecil lebih ramah bagi jaringan saraf.

---

## Pengaturan Kami

| Pilihan | Nilai | Mengapa |
|--------|-------|-----|
| Lingkungan | `CartPole-v1` | Status 4-D, 2 tindakan — mudah dimodelkan |
| Data | 20.000 transisi dari kebijakan acak | Cakupan luas dari ruang status |
| Jaringan | MLP, 2 × 128 ReLU tersembunyi | MLP = Multi-Layer Perceptron (jaringan saraf standar). Dua lapisan tersembunyi masing-masing 128 neuron menggunakan aktivasi ReLU. Kapasitas cukup, cepat untuk dilatih. |
| Loss | MSE pada `(delta_status, imbalan)` | MSE = Mean Squared Error (rata-rata kesalahan prediksi kuadrat). Loss regresi standar. |
| Pengoptimal | Adam, lr = 1e-3, 30 epoch | Adam = pengoptimal adaptif (menyesuaikan tingkat pembelajaran per parameter secara otomatis). Langsung pakai berarti tidak perlu penyetelan khusus. |

Seluruh pelatihan selesai dalam beberapa detik di CPU.

---

## Seperti Apa Hasil yang "Bagus"?

Dua diagnostik sangat penting:

### 1. Akurasi satu langkah (MSE validasi)

Ini adalah "seberapa baik model memprediksi SATU langkah ke depan?" Setelah 30 epoch Anda akan melihat MSE validasi di kisaran **1e-4 hingga 1e-3**. Itu sangat kecil — sudut tiang dan posisi kereta akurat hingga beberapa angka desimal.

### 2. **Compounding error** pada rollout k-langkah

Ini adalah tes yang *sebenarnya*. Ambil sebuah status, masukkan ke dalam model, lalu ambil prediksinya dan masukkan kembali ke dalam model — selama `k` langkah berturut-turut. Error akan tumbuh karena setiap langkah menambahkan sedikit kebisingan di atas prediksi sebelumnya.

```
Langkah 1:  L2 error ≈ 0,01 (hampir sempurna)
Langkah 5:  L2 error ≈ 0,05
Langkah 10: L2 error ≈ 0,15
Langkah 20: L2 error ≈ 0,40 (terlihat melenceng)
```

*(L2 error = jarak Euclidean antara status berikutnya yang diprediksi dan yang asli — bayangkan sebagai "seberapa jauh tebakan model dalam ruang status 4-D?")*

**Mengapa ini penting.** Jika kita merencanakan 15 langkah ke depan dengan model, status *tepat* pada langkah 15 akan salah — tetapi jika peringkat relatif "rencana baik vs rencana buruk" tetap terjaga, perencanaan masih berfungsi. (Inilah yang dimanfaatkan oleh `model_based_planning.py`.)

Plot di `outputs/world_model.png` menunjukkan kedua diagnostik tersebut berdampingan: kurva training-loss turun dengan baik pada skala log, dan kurva rollout-error naik dengan semestinya.

---

## Mengapa Memprediksi *Delta*?

Bandingkan dua cara merumuskan masalah yang sama ke jaringan:

| Target | Magnitudo tipikal | Mudah atau sulit? |
|--------|------------------:|--------------|
| `status_berikutnya` | 0–2,4 (posisi krt) | Jaringan harus mereproduksi posisi **DAN** perubahan kecilnya |
| `status_berikutnya - status` | ~0,02 | Jaringan hanya mempelajari perubahan kecilnya |

Memprediksi delta juga berarti: jika jaringan mengeluarkan angka nol (seperti yang sering dilakukan jaringan pemula yang belum terlatih), prediksinya hanyalah "tidak ada yang bergerak" — default yang masuk akal dan aman untuk satu langkah waktu. Sebaliknya, memprediksi `status_berikutnya` secara absolut secara langsung pada awalnya akan mengeluarkan nilai sampah yang sepenuhnya acak, menyebabkan pelatihan awal menjadi sangat tidak stabil.

---

## Apa yang Kita Dapatkan dari Ini

World model yang terlatih adalah fondasi untuk:

- **Perencanaan (Planning)** — mencari melalui urutan tindakan imajiner (lihat `model_based_planning.py`).
- **Augmentasi gaya Dyna** — melatih jaringan-Q pada data imajiner untuk melipatgandakan efisiensi sampel.
- **Keingintahuan / eksplorasi** — mengunjungi status yang tidak dapat diprediksi model dengan baik.
- **Makalah Dreamer / World-Models** — melatih *kebijakan* sepenuhnya di dalam model dengan nol interaksi dunia nyata selain pengumpulan data awal.

---

## Batasan dan Peringatan

- **Hanyutan di luar distribusi (Out-of-distribution drift).** Model hanya mengetahui bagian dunia yang telah dilihatnya. Rencanakan terlalu agresif dan Anda akan berakhir di wilayah yang belum pernah dikunjungi model — prediksi di sana murni fantasi.
- **Compounding error.** Merencanakan melampaui **cakrawala (horizon)** yang panjang (banyak langkah ke depan) tidak dapat diandalkan karena kesalahan yang menumpuk, seperti yang ditunjukkan grafik. Sistem modern mengatasi hal ini dengan menggunakan **probabilistic ensembles** (melatih beberapa model dan memeriksa apakah mereka setuju, seperti dalam PETS atau Dreamer) sehingga perencana tahu persis *seberapa tidak yakin* model tersebut di setiap langkah dan dapat menghindari jalur berisiko yang tidak diketahui.
- **Lingkungan stokastik.** Regressor deterministik standar hanya memprediksi hasil *rata-rata* dan sama sekali melewatkan penyebaran hasil yang mungkin. Lingkungan dunia nyata yang kompleks membutuhkan model probabilistik (seperti model dengan output Gaussian, atau **latent stochastic models** — jaringan yang mengodekan status dunia sebagai distribusi probabilitas dalam ruang terkompresi, membiarkannya menangkap keacakan yang asli alih-alih merata-ratakannya) untuk merepresentasikan ketidakpastian dan keacakan secara akurat.

---

## Kata Kunci

| Istilah | Bahasa Sederhana |
|------|---------------|
| **World model** | Jaringan saraf yang meniru lingkungan |
| **Dinamika (Dynamics)** | Fungsi `(s, a) → s'` |
| **Reward model** | Fungsi `(s, a) → r` (sering kali digabungkan) |
| **Prediksi satu langkah** | Apa yang dikeluarkan model dari status nyata |
| **Rollout** | Prediksi satu langkah yang diulang, dengan memasukkan kembali outputnya |
| **Compounding error** | Kesalahan kecil yang bertumbuh selama rollout |

---

## Ringkasan Satu Kalimat

> **World model adalah salinan saraf kecil dari alam semesta yang dapat dikonsultasikan — dan dimimpikan di dalamnya — oleh agen sebelum mempertaruhkan tindakan nyata.**

Berikutnya: `model_based_planning.py` menggunakan model ini untuk pengambilan keputusan yang sebenarnya.
