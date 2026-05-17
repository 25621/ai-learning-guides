# A2C: Sang Aktor dan Kritikus Bekerja Bersama

## Ide Besar

REINFORCE menunggu sampai permainan benar-benar selesai sebelum melakukan pembaruan. Itu seperti seorang pelatih yang menonton seluruh pertandingan sepak bola dalam diam, lalu memberikan semua umpan balik di akhir.

**A2C (Advantage Actor-Critic)** memberikan umpan balik SELAMA permainan — setiap beberapa langkah, pelatih berhenti sejenak untuk mengatakan: "Operan itu bagus! Tekel itu buruk!"

Ini jauh lebih cepat dan lebih efisien.

---

## Mengenal Dua Karakter

> **Apa itu LunarLander?** Di seluruh dokumen ini kita menggunakan lingkungan **LunarLander** — simulasi fisika di mana Anda mengendalikan pesawat ruang angkasa kecil dan harus mendaratkannya dengan lembut di landasan target di bulan menggunakan tiga mesin (kiri, utama, dan kanan). Ini adalah tolok ukur standar dalam pembelajaran penguatan, tersedia di pustaka Gymnasium.

### Sang Aktor (Actor) 🎭
**Aktor** adalah kebijakan (policy) — ia memutuskan tindakan mana yang akan diambil.

> "Saya berada di status ini. Haruskah saya menyalakan mesin kiri atau mesin kanan?"

**Contoh kehidupan nyata:** *Pengemudi* mobil yang memutar setir dan menginjak pedal.

### Sang Kritikus (Critic) 🎬
**Kritikus** memperkirakan seberapa baik situasi saat ini — nilai V(s).

> "Berada di status INI bernilai sekitar +150 poin dari total imbalan masa depan."

**Contoh kehidupan nyata:** *Navigator* yang duduk di sebelah pengemudi, berkata "Kita berada di jalan yang bagus — diperkirakan akan tiba dalam 30 menit." atau "Kita sedang menuju kemacetan — ini akan lambat."

### Mereka Berbagi Otak
Dalam implementasi kami, keduanya menggunakan **tulang punggung jaringan saraf yang sama**:

```
          Status (8 angka untuk LunarLander)
                       ↓
          ┌─────────────────────────┐
          │  Lapisan Bersama        │
          │  [256 neuron] → ReLU   │
          │  [256 neuron] → ReLU   │
          └────────┬────────┬───────┘
                   ↓        ↓
          Kepala Aktor    Kepala Kritikus
          [4 output]     [1 output]
          (prob aksi)    (V(s))
```

- **ReLU** (Rectified Linear Unit): fungsi aktivasi yang diterapkan setelah setiap lapisan — ia mengeluarkan `max(0, x)`, mempertahankan nilai positif dan menjadikan nol untuk nilai negatif. Ini memungkinkan jaringan mempelajari pola non-linear.
- **prob aksi**: probabilitas pengambilan masing-masing dari 4 tindakan. Aktor mengambil sampel dari distribusi ini untuk memilih tindakan di setiap langkah.

**Contoh kehidupan nyata:** Satu otak, dua pekerjaan — seperti sopir taksi yang mengemudi (aktor) DAN tahu apakah rutenya bagus (kritikus). Berbagi otak berarti belajar lebih cepat!

---

## Advantage: Apakah Ini Lebih Baik Dari Perkiraan?

Sama seperti REINFORCE dengan baseline, A2C menghitung **Advantage**:

> A(s, a) = "Hasil aktual" − "Apa yang kita harapkan"

Tetapi di sini, "hasil aktual" berasal dari **n-step bootstrap** Kritikus (**bootstrapping** = menggunakan prediksi Kritikus sendiri V(s) untuk memperkirakan nilai langkah-langkah masa depan, alih-alih menunggu episode aktual berakhir — seperti memperkirakan nilai ujian akhir Anda di tengah semester menggunakan nilai Anda saat ini):

```
Return TD aktual: r_t + γ · r_{t+1} + γ² · r_{t+2} + ... + γⁿ · V(s_{t+n})
Advantage A_t = Return TD - V(s_t)
```

**Contoh kehidupan nyata:** Anda berharap mencetak 3 gol dalam pertandingan ini (V(s)). Jika Anda mencetak 5 gol, advantage Anda adalah +2. Jika Anda mencetak 1 gol, advantage Anda adalah -2.

Advantage positif → "tindakan itu membantu lebih dari yang diharapkan → lakukan lebih sering!"
Advantage negatif → "tindakan itu membantu kurang dari yang diharapkan → lakukan lebih jarang!"

---

## Mengapa Menggunakan Beberapa Lingkungan Paralel?

A2C kami menggunakan **8 salinan** LunarLander yang berjalan pada saat yang sama!

**Mengapa?** Karena pengalaman dari satu lingkungan **berkorelasi** — satu langkah mengikuti langkah sebelumnya dengan erat. Korelasi ini menipu jaringan saraf sehingga menganggap pola lebih andal daripada yang sebenarnya.

Dengan 8 lingkungan, setiap langkah memberikan 8 pengalaman independen dari situasi yang sangat berbeda. Ini memutus korelasi dan menstabilkan pelatihan secara dramatis.

**Contoh kehidupan nyata:** Untuk belajar tentang cuaca, mana yang lebih baik:
- Menonton satu kota selama 8 jam berturut-turut (berkorelasi — jika cerah jam 2 siang, kemungkinan besar cerah jam 3 sore)
- Menonton 8 kota secara bersamaan (tidak berkorelasi — pola cuaca berbeda, informasi lebih banyak!)

```
Lingkungan 1: [mendarat di bulan, nyalakan kiri, jatuh, reset...]
Lingkungan 2: [jatuh terlalu cepat, nyalakan keduanya, melayang, mendarat...]
Lingkungan 3: [miring ke kanan, nyalakan kanan, stabilkan, mendarat...]
...
Lingkungan 8: [hanyut ke kiri, nyalakan kiri, stabil, ...]
```

Kedelapan lingkungan memperbarui jaringan secara bersamaan — 8× lebih banyak pengalaman beragam per pembaruan!

---

## Pembaruan N-Step: Jangan Tunggu Permainan Berakhir

REINFORCE menunggu satu episode penuh (bisa 1000 langkah!).

A2C diperbarui setiap **n_steps = 128 langkah**:

```
Mainkan 128 langkah di 8 lingkungan
    → Dapatkan 128 × 8 = 1024 tupel pengalaman
    → Hitung advantage dan return
    → Perbarui Aktor dan Kritikus
    → Mainkan 128 langkah lagi...
```

**Contoh kehidupan nyata:** Seorang siswa yang belajar untuk ujian.
- Gaya REINFORCE: Baca seluruh buku teks, KEMUDIAN ikuti tes latihan.
- Gaya A2C: Baca 10 halaman, ikuti kuis, baca 10 halaman lagi, ikuti kuis...

Umpan balik yang lebih sering = belajar lebih cepat!

---

## Tiga Loss Gabungan

A2C dilatih dengan tiga istilah loss secara bersamaan:

**Loss** adalah angka yang coba diminimalkan oleh pengoptimal. Loss yang lebih kecil berarti perilaku jaringan saat ini lebih dekat dengan tujuan pelatihan.

### 1. Actor Loss (Policy Gradient)
Membuat tindakan yang menguntungkan (advantageous) lebih mungkin terjadi:
```
L_actor = -E[log π(a|s) · A(s,a)]
```
Jika A > 0: tingkatkan probabilitas tindakan tersebut
Jika A < 0: kurangi probabilitas tindakan tersebut

### 2. Critic Loss (Value Function MSE)
Membuat prediksi nilai lebih akurat (**MSE** = Mean Squared Error: kuadratkan kesalahan prediksi dan rata-ratakan — penguadratan memberikan penalti lebih berat pada kesalahan besar daripada kesalahan kecil):
```
L_critic = E[(V(s) - return)²]
```
Seperti melatih model **regresi** apa pun (regresi = memprediksi angka kontinu, di sini return yang diharapkan V(s)) — minimalkan kesalahan prediksi.

### 3. Bonus Entropi (Eksplorasi)
Mencegah kebijakan menjadi terlalu percaya diri terlalu cepat:
```
L_entropy = -H[π(·|s)] = E[log π(a|s)]
```
Entropi tinggi = pilihan tindakan yang beragam = eksplorasi
Entropi rendah = pilihan yang percaya diri dan sempit = eksploitasi

**Contoh kehidupan nyata:** Bonus entropi seperti seorang guru yang berkata "Jangan hanya menebak A pada setiap pertanyaan pilihan ganda! Cobalah jawaban yang berbeda agar Anda belajar apa yang berhasil."

```
Total loss = L_actor + 0.5 × L_critic - 0.01 × entropi
```

---

## LunarLander: Tantangan yang Lebih Sulit

**LunarLander-v3** adalah lingkungan Gymnasium (sebelumnya OpenAI Gym) — "v3" adalah nomor versi yang menunjukkan revisi ketiga dari lingkungan ini. Agen mengendalikan pesawat ruang angkasa kecil yang harus mendarat dengan aman di landasan yang ditentukan di bulan. Ini jauh lebih sulit daripada CartPole:
- Ruang status 8-dimensi (posisi, kecepatan, sudut, kontak kaki, bahan bakar)
- 4 tindakan diskret (tidak melakukan apa pun, nyalakan kiri, nyalakan utama, nyalakan kanan)
- Imbalan: +100 untuk mendarat, -100 untuk jatuh, penalti bahan bakar kecil

Kurva pelatihan menunjukkan peningkatan bertahap dari imbalan yang sangat negatif menuju imbalan positif. A2C di LunarLander membutuhkan pengalaman yang signifikan sebelum pendarat mempelajari stabilitas dasar.

---

## Persamaan Utama

```
return n-step:  G_t = r_t + γ·r_{t+1} + ... + γⁿ·V(s_{t+n})
Advantage:      A_t = G_t - V(s_t)
Pembaruan Aktor:   θ_π ← θ_π - α ∇ L_actor
Pembaruan Kritikus:  θ_V ← θ_V - α ∇ L_critic
```

---

## Poin-Poin Penting

| Konsep | Bahasa Sederhana |
|---------|---------------|
| **Aktor** | Kebijakan — memutuskan apa yang harus dilakukan |
| **Kritikus** | Fungsi nilai — menilai seberapa baik situasinya |
| **Advantage** | "Apakah ini lebih baik dari yang diharapkan?" (aktual - diharapkan) |
| **Return n-step** | Lihat n langkah ke depan sebelum melakukan bootstrap dengan V(s) |
| **Lingkungan paralel** | Beberapa lingkungan untuk pengalaman yang tidak berkorelasi dan beragam |
| **Bonus entropi** | Mendorong aktor untuk terus mencoba hal-hal baru |

---

## Apa Selanjutnya?

A2C hebat tetapi memiliki satu kelemahan utama: terkadang ia memperbarui kebijakan terlalu agresif. Satu pembaruan buruk dapat menghancurkan semua pembelajaran baik di pembaruan sebelumnya.

**PPO (Proximal Policy Optimization)** memperbaiki hal ini dengan "klip pengaman" cerdas yang mencegah pembaruan tunggal mengubah kebijakan terlalu banyak.

Lihat `ppo_scratch.py` untuk implementasi PPO!
