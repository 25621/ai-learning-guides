# Menggunakan Model yang Dipelajari untuk Perencanaan (MPC) 🔮

## Ide Besar {#the-big-idea}

Anda memiliki sebuah **world model** (jaringan saraf yang memprediksi masa depan). Sekarang apa?

Penggunaan yang paling langsung adalah **perencanaan (planning)**: di setiap saat, tanyakan pada model "apa yang akan terjadi jika saya mencoba rencana *ini*? rencana *itu*? rencana *lainnya*?" Lalu lakukan rencana mana pun yang terlihat paling baik — tetapi **hanya langkah pertamanya saja**. Karena model tersebut tidak sempurna, kita hanya menjalankan satu tindakan, mengamati status baru yang sebenarnya dari lingkungan nyata, dan kemudian merencanakan ulang dari awal.

Trik ini memiliki nama: **Model Predictive Control** (MPC).

---

## Analogi Kehidupan Nyata {#a-real-life-analogy}

Anda berada di restoran sedang melihat menu. Anda tidak berkomitmen pada pesanan lima menu sekaligus di tempat — Anda memesan menu pertama, melihat seberapa kenyang Anda, lalu memutuskan kembali untuk hidangan penutup.

Atau: Anda sedang mengemudi di jalan yang berkelok-kelok. Anda tidak mengunci input penyetiran untuk 30 detik ke depan — Anda terus-menerus melihat ke depan, merencanakan beberapa detik ke depan, mengambil tindakan penyetiran berikutnya, dan merencanakan ulang.

Loop **rencana-jauh / bertindak-dekat / rencana-ulang** itu adalah MPC.

---

## Cara Kerja "Random Shooting" {#how-random-shooting-works}

Ada perencana yang lebih canggih — misalnya:
- **CEM** (Cross-Entropy Method): memperbaiki distribusi atas rencana secara iteratif dengan hanya menyimpan k rencana terbaik di setiap putaran.
- **MCTS** (Monte Carlo Tree Search): membangun pohon pencarian yang dipandu oleh statistik simulasi, digunakan oleh AlphaGo dan MuZero.
- **Perencana berbasis gradien (Gradient-based planners)**: membedakan prediksi model sehubungan dengan tindakan dan mengikuti gradien secara langsung.

Kita menggunakan yang paling sederhana yang berhasil: **random shooting**.

```
Mengingat status s saat ini:
    1. Ambil sampel N=200 urutan tindakan acak dengan panjang H=15.
    2. Untuk setiap urutan, simulasikan melalui world model dari s, menjumlahkan
       imbalan yang dibentuk di setiap langkah. (200 mimpi secara paralel — cepat!)
    3. Temukan urutan dengan total imbalan prediksi tertinggi.
    4. Jalankan tindakan PERTAMA dari urutan tersebut di lingkungan nyata.
    5. Amati status berikutnya yang nyata. Buang sisa rencana lainnya.
    6. Kembali ke langkah 1 — rencanakan ulang dari awal.
```

200 rencana × 15 langkah = 3.000 transisi imajiner per langkah nyata. World model menjalankan semuanya dalam satu forward pass jaringan saraf yang di-batch — biasanya beberapa milidetik.

---

## Mengapa Merencanakan Ulang Setiap Langkah? {#why-re-plan-every-step}

Karena model tersebut tidak sempurna. Kesalahan menumpuk selama proses rollout (seperti yang terlihat dalam grafik yang dihasilkan oleh `world_model.py`, disimpan di `outputs/world_model.png`). Rencana pada langkah 0 hanya dapat diandalkan untuk beberapa langkah pertama; pada langkah 15 model tersebut sudah berhalusinasi. Jadi kita hanya mempercayai **langkah pertama**, lalu menyegarkan rencana dengan status nyata terbaru.

Ini adalah alasan yang sama mengapa manusia tidak menulis rencana catur 100 langkah dan terpaku padanya — keadaan berubah, dan semakin jauh Anda menebak, semakin tidak sesuai dengan kenyataan.

---

## Sebuah Masalah: Imbalan Harus Memberi Tahu Sesuatu kepada Perencana {#a-wrinkle-the-reward-has-to-tell-the-planner-something}

Di CartPole, imbalan nyatanya adalah `+1` setiap langkah sampai tiang jatuh. Model akan dengan setia memprediksi `+1, +1, +1, ...` untuk hampir setiap rencana, karena rencana acak jarang berakhir cepat di dalam model — sehingga setiap rencana mendapat skor yang sama. Perencana tidak memiliki dasar untuk memilih.

Solusinya: ganti imbalan nyata dengan **proksi halus (smooth proxy)** selama perencanaan:

```python
reward_proxy(state) = 1
                    - |sudut_tiang| / 0.21          # tiang tegak? (1=ya)
                    - 0.1 * |posisi_kereta| / 2.4  # kereta di tengah? (1=ya)
```

Sekarang rencana yang *akan* berakhir dengan tiang jatuh mendapatkan skor yang jauh lebih buruk daripada rencana yang tetap tegak. Perencana dapat mengurutkannya.

> **Pelajaran kehidupan nyata.** Sinyal imbalan yang datar — "Anda bertahan satu detik lagi" — tidak berguna untuk perencanaan jangka pendek. Sinyal yang padat dan dibentuk (shaped) sangat membantu.

---

## Apa yang Dilakukan Kode Kami {#what-our-code-does}

`model_based_planning.py`:

1. **Memuat** bobot world model yang disimpan oleh `world_model.py`. (Jika tidak ada, ia melatih ulang satu secara langsung.)
2. **Menjalankan 20 episode** MPC pada CartPole-v1 yang sebenarnya.
3. **Juga menjalankan 20 episode** dengan kebijakan acak seragam, sebagai baseline.
4. **Membuat plot** keduanya berdampingan dan mencetak rata-ratanya.

### Apa yang seharusnya Anda lihat saat menjalankannya {#what-you-should-see-when-you-run-it}

| Kebijakan | Rata-rata imbalan (langkah bertahan) |
|--------|-------------------------------:|
| Acak          | ~22 (tipikal untuk CartPole — tiang jatuh cepat) |
| MPC (milik kita) | ~150–500 (bervariasi menurut seed; banyak episode mendekati 500) |
| Maks mungkin  | 500 |

Peningkatan **5–25×** tersebut dicapai tanpa jaringan kebijakan, tanpa fungsi nilai, dan tanpa pelatihan lebih lanjut. Hanya sebuah world model + 200 mimpi per langkah.

Plot `outputs/model_based_planning.png` menunjukkan dua batang berwarna per episode — MPC hampir selalu lebih tinggi daripada Acak, dan banyak episode mencapai batas atas 500 langkah.

---

## Kelebihan Perencanaan Berbasis Model {#strengths-of-model-based-planning}

- **Efisien sampel.** Semua pembelajaran dilakukan dari satu batch transisi acak. Tidak diperlukan interaksi lingkungan lebih lanjut untuk menghasilkan kebijakan yang berguna.
- **Mudah diubah tujuannya.** Ingin mengontrol agen secara berbeda? Ubah proksi imbalannya — tidak perlu pelatihan ulang. (Coba maksimalkan kecepatan kereta untuk bersenang-senang.)
- **Dapat diinterpretasikan.** Anda dapat memeriksa rencana yang dipertimbangkan agen, lintasan yang diprediksi, dan skornya.

## Kelemahan (dan Apa yang Dilakukan Orang untuk Mengatasinya) {#weaknesses-and-what-people-do-about-them}

- **Random shooting itu bodoh.** Ia mengambil sampel rencana secara buta. Untuk dimensi yang lebih tinggi, beralihlah ke **CEM** (Cross-Entropy Method — lihat di atas) atau **iLQR** (Iterative Linear-Quadratic Regulator, metode kontrol-optimal klasik yang mendekati model sebagai linear lokal dan menyelesaikannya secara analitik) atau perencana **berbasis gradien** penuh yang memperbaiki tindakan dengan mengikuti gradien melalui model yang dapat dideferensiasi.
- **Kesalahan model yang menumpuk.** Cakrawala panjang akan menyimpang. Orang-orang menggunakan **probabilistic ensembles** (beberapa model yang dilatih pada data yang sama, seperti dalam PETS, Chua et al. 2018) sehingga perencana dapat menyadari ketidaksepakatan dan memberikan penalti pada rencana yang modelnya tidak yakin.
- **Imbalan nyata adalah apa yang kita inginkan, pada akhirnya.** Pembentukan imbalan (reward shaping) membantu, tetapi untuk tugas yang lebih kompleks orang mempelajari **fungsi nilai (value function)** yang dilatih *di dalam* world model — kritikus terpelajar yang memperkirakan return jangka panjang dari status mana pun tanpa memerlukan proksi buatan tangan. Baik **Dreamer** (yang melatih aktor-kritikus sepenuhnya dalam imajinasi laten) maupun **MuZero** (yang memadukan MCTS dengan jaringan nilai terpelajar) menggunakan ide ini.

---

## Bagaimana Ini Terhubung dengan Sistem Modern {#how-this-connects-to-modern-systems}

Resep persis yang baru saja Anda jalankan — **dinamika yang dipelajari + perencanaan** — adalah tulang punggung dari beberapa sistem RL terkuat dalam riset AI modern:

- **MuZero** (DeepMind): menggabungkan world model yang dipelajari dengan Monte Carlo Tree Search. Ia menguasai Go, catur, shogi, dan Atari tanpa memerlukan aturan sebelumnya.
- **Dreamer / DreamerV3** (Hafner et al.): melatih kebijakan *di dalam* world model **latent-space** yang dipelajari (artinya model mengompresi gambar atau status mentah menjadi representasi abstrak yang kompak sebelum memprediksi masa depan). Ia mencapai performa mutakhir (top-tier) di lebih dari 100 benchmark.
- **PETS / PlaNet / TD-MPC**: ini adalah keluarga algoritma (lini riset) yang menyisipkan ide ini secara tepat ke tugas kontrol kontinu yang kompleks seperti robotika.

Anda telah membangun — dalam beberapa ratus baris — anggota terkecil dari keluarga tersebut.

---

## Kata Kunci {#key-words}

| Istilah | Bahasa Sederhana |
|------|---------------|
| **MPC**           | Model Predictive Control — rencanakan ke depan, bertindak sekali, rencanakan ulang |
| **Random shooting** | Beri skor pada banyak rencana acak, pilih yang terbaik |
| **Cakrawala / Horizon (H)** | Berapa langkah rencana melihat ke depan |
| **N sampel**     | Berapa banyak kandidat rencana yang kita pertimbangkan per langkah |
| **Receding horizon** | Merencanakan ulang pada setiap langkah alih-alih berkomitmen pada satu rencana |
| **Proksi imbalan / pembentukan imbalan (Reward proxy / shaping)** | Imbalan pengganti yang halus yang memberi perencana sinyal yang berguna untuk dioptimalkan |

---

## Ringkasan Satu Kalimat {#one-sentence-summary}

> **Setelah Anda memiliki world model, perencanaan hanyalah "memimpikan seratus masa depan, pilih langkah pertama yang terbaik, ulangi."**

Itulah rahasia utuh dari RL berbasis model.
