# Behavioral Cloning (BC) 🐒

## Apa Itu?

Bayangkan Anda ingin belajar bermain tenis. Anda menonton ratusan jam rekaman
pertandingan Wimbledon dan cukup **meniru apa yang dilakukan para pemain**. Anda
tidak memikirkan apakah pukulan mereka adalah pukulan *terbaik* — Anda hanya
menyesuaikan posisi tubuh Anda dengan posisi mereka dan mengayunkan raket dengan cara yang sama.

Itulah behavioral cloning. **Tanpa imbalan. Tanpa perencanaan. Hanya imitasi.**

Dalam istilah RL: ambil dataset pasangan `(status, tindakan)` dan latih
jaringan saraf untuk memprediksi tindakan dari status, persis seperti
model klasifikasi gambar memprediksi kucing-vs-anjing. "Label" adalah
tindakan apa pun yang diambil oleh pengumpul data.

---

## Perbedaannya dengan RL Offline yang "Asli"

| Pendekatan | Menggunakan imbalan? | Bisa mengalahkan data? |
|----------|---------------|---------------------|
| **BC**   | ❌ tidak       | ❌ tidak — paling banter, menyamai kualitas rata-rata data |
| **CQL** (dan kawan-kawan) | ✅ ya | ✅ ya — bisa merangkai transisi yang baik dari data campuran |

BC adalah "sudut pandang pembelajaran terawasi (supervised learning)" dari RL. Ini sangat sederhana, seringkali
secara mengejutkan kuat, dan merupakan baseline universal. **Jika algoritma
RL offline tidak bisa mengalahkan BC pada dataset yang sama, berarti algoritma itu belum melakukan apa pun.**

---

## Contoh Kehidupan Nyata

- **Belajar mengemudi dari rekaman dashcam.** Lihat ke jalan, prediksi
  sudut setir yang digunakan manusia. Dua contoh bersejarah:
  - **ALVINN (1989)** — pengemudi jaringan saraf pertama; jaringan
    3-lapisan kecil yang dilatih pada input kamera + laser untuk mengemudikan van di jalan raya.
  - **NVIDIA PilotNet (2016)** — CNN dalam modern yang dilatih secara end-to-end pada
    rekaman dashcam; mempelajari pemeliharaan jalur dan penyetiran dasar murni dengan
    meniru pengemudi manusia, tanpa aturan buatan tangan.
- **Magang yang meniru koki ahli.** "Apa pun yang dilakukan koki, saya lakukan."
  Berhasil jika kokinya hebat; menghasilkan koki yang buruk jika kokinya buruk.
- **GitHub Copilot.** Auto-complete dilatih untuk memprediksi "kode apa
  yang akan diketik manusia selanjutnya?" — murni imitasi log kode sumber.
- **Meniru kakak Anda.** Anak-anak melakukan ini selama bertahun-tahun sebelum mereka
  mulai memikirkan *mengapa* kakak mereka melakukan apa yang mereka lakukan.

---

## Matematika (Satu Baris)

Untuk setiap `(s, a)` dalam dataset, minimalkan:

```
loss = -log π(a | s)        (cross-entropy)
```

Itu saja. Kebijakan `π` hanyalah sebuah MLP yang mengeluarkan logit tindakan;
pelatihannya identik dengan MNIST. Mari kita bedah istilah-istilahnya:
- **`π` (Pi):** Simbol standar untuk "kebijakan" (policy) — aturan atau jaringan saraf yang memutuskan apa yang harus dilakukan.
- **MLP (Multi-Layer Perceptron):** Jaringan saraf dasar yang standar.
- **Logit:** Skor mentah yang belum dinormalisasi yang dikeluarkan jaringan sebelum kita mengubahnya menjadi probabilitas.
- **Cross-entropy:** Formula standar untuk memberikan penalti pada model saat model tersebut memberikan probabilitas rendah pada jawaban yang benar.
- **MNIST:** Dataset pemula yang terkenal berisi digit tulisan tangan.

Melatih agen untuk memainkan game melalui BC secara harfiah identik dengan melatih jaringan untuk mengenali digit tulisan tangan di MNIST. Di MNIST, inputnya adalah gambar dan outputnya adalah digit (0-9). Di BC, inputnya adalah status game dan outputnya adalah tindakan (misalnya, "bergerak ke kiri").

---

## Apa yang Dilakukan Kode Kami

Skrip `behavioral_cloning.py`:

1. **Memuat keempat dataset** yang dibuat oleh `d4rl_dataset.py`
   (`random`, `medium`, `expert`, `medium-replay`).
2. Untuk setiap dataset, **melatih kebijakan BC yang terpisah** selama 10.000 langkah gradien
   cross-entropy. Kolom imbalan (reward) sepenuhnya diabaikan.
3. Setiap 2.500 langkah, **mengevaluasi** kebijakan saat ini dengan menjalankannya secara
   rakus (greedy) di lingkungan CartPole-v1 yang sebenarnya (20 episode, dirata-ratakan).
4. Membuat plot:
   - Diagram batang: return BC akhir per dataset.
   - Grafik kurva pembelajaran: bagaimana setiap varian BC naik selama pelatihan.

---

## Apa yang Seharusnya Anda Lihat

Jalannya program biasanya mencetak:

```
Hasil evaluasi akhir:
  BC on random          ->    ~20  ± sedikit   (≈ bermain acak)
  BC on medium          ->   ~150  ± besar     (≈ kebijakan medium)
  BC on expert          ->   ~480  ± kecil     (≈ kebijakan pakar)
  BC on medium-replay   ->    ~60  ± besar     (≈ RATA-RATA data campuran)
```

Plot batang membuat ceritanya jelas: **return BC mengikuti return rata-rata dataset.**
Ia tidak bisa melampaui batas atas (ceiling) tersebut karena ia tidak punya cara untuk
memilih bagian yang "baik" dari dataset campuran daripada bagian yang "buruk" — keduanya
adalah target imitasi yang sama validnya.

Itulah intinya: **BC mewarisi batas atas data.**

---

## BC vs CQL — Perbandingan Paling Jelas

Pada dataset **medium-replay** (kasus paling realistis dengan kualitas campuran):

| Metode | Perkiraan return akhir | Mengapa? |
|--------|--------------------:|------|
| BC     | ~60   | Meniru *rata-rata* dari percobaan awal yang gagal + yang bagus kemudian |
| CQL    | ~400+ | Menggunakan imbalan untuk memilih transisi Q tinggi; merangkai kebijakan yang baik dari data campuran |

Jadi CQL **mengalahkan data**, BC **menyamai data**. Itulah alasan mengapa
RL offline adalah bidang penelitian dan bukan sekadar "melakukan pembelajaran imitasi".
Ketika data memiliki kualitas campuran (seperti log nyata pada umumnya),
metode yang sadar imbalan (reward-aware) mendapatkan hasil lebih banyak.

Pada data **expert**, perbandingannya berbalik: BC menyamai pakar (~480). Anda mungkin bertanya-tanya mengapa CQL "imbang" di sini daripada kalah. Karena CQL dirancang untuk bersikap *konservatif* dan menghukum tindakan yang tidak terlihat dalam dataset, ia akhirnya melakukan persis seperti apa yang dilakukan pakar. Ia tidak bisa mengalahkan pakar (karena skor maksimum yang mungkin sudah tercapai), tetapi ia juga tidak secara aktif merusak strategi pakar. Ia hanya imbang dengan performa BC.

Ini adalah trade-off terkenal antara "kualitas data vs algoritma":

```
                            Data PAKAR  →  BC menang, CQL imbang
   Kecanggihan algoritma     ↑         
                            Data CAMPURAN →  CQL jelas mengalahkan BC
                            
                            Data ACAK   →  Semua gagal; butuh eksplorasi
```

---

## Di Mana BC Digunakan dalam RL Modern

- **Pra-pelatihan untuk RL online.** Banyak sistem modern (RT-1, Voyager,
  bot permainan game) dimulai dengan BC pada demonstrasi, lalu disesuaikan (fine-tune)
  online dengan PPO/SAC.
- **RLHF.** Langkah 1 dari InstructGPT adalah fine-tuning terawasi — BC murni pada
  tanggapan yang ditulis manusia. PPO + model imbalan menyusul kemudian.
- **DAgger (Ross et al., 2011).** Ekstensi cerdas untuk memperbaiki masalah **kesalahan beruntun (compounding-error)**.
  *Mengapa kesalahan beruntun menjadi masalah jika BC meniru dengan sempurna?* Bahkan jika model BC 99% akurat, kesalahan 1% itu akhirnya terjadi. Saat itu terjadi, agen memasuki status yang belum pernah ia lihat dalam dataset yang dikemudikan dengan sempurna. Karena bingung, ia membuat kesalahan yang lebih besar, bergerak lebih jauh dari data yang diketahui, hingga berakhir dengan kegagalan total (seperti mengemudi ke jurang).
  *Solusinya:* Kita bisa meminta pakar untuk mengemudi selamanya, tetapi waktu pakar itu mahal. Sebaliknya, DAgger membiarkan kebijakan BC mengemudi. Saat kebijakan membuat kesalahan dan hanyut ke status yang aneh, kita berhenti sejenak, bertanya kepada pakar "apa yang akan Anda lakukan *tepat di sini*?", dan menambahkannya ke dataset. Kita hanya "bertanya kembali kepada pakar pada status yang dikunjungi kebijakan BC" karena kita hanya butuh pakar untuk mengajari kita cara pulih dari kesalahan spesifik kita sendiri, daripada selalu bertanya kepada mereka.
- **Decision Transformer (Chen et al., 2021).** BC "pintar" yang
  mengkondisikan prediksi tindakan pada *return-to-go* yang diinginkan,
  secara esensial mengubah RL offline kembali menjadi prediksi token berikutnya.

---

## Kata Kunci untuk Diingat

| Kata | Makna |
|------|---------|
| **Imitation learning** | Istilah payung untuk "tiru sang pendemo"; BC adalah anggota paling sederhana |
| **Compounding error** | Kesalahan BC kecil membawa Anda ke status yang tidak pernah dilihat dataset, di mana kesalahan menumpuk |
| **Demonstration data** | Trajektori yang dihasilkan oleh pakar, digunakan sebagai set pelatihan BC |
| **Data ceiling** | Return BC dibatasi oleh return rata-rata dalam dataset |
| **DAgger** | Solusi interaktif untuk kesalahan beruntun |

---

## Ringkasan Satu Kalimat

> **Behavioral cloning hanyalah pembelajaran terawasi pada pasangan (status, tindakan) — kuat saat datanya bagus, tidak berdaya saat datanya campuran.**
