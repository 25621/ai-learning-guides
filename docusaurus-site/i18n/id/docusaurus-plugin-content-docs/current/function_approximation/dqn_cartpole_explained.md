# Deep Q-Network (DQN) dari Nol 🧠

## Masalah dengan Linear

Ingat rumus linear kita sebelumnya?

> Skor = w₁ × posisi_kereta + w₂ × kecepatan_kereta + w₃ × sudut_tiang + w₄ × kecepatan_tiang

Ini bekerja cukup baik untuk CartPole, tetapi bagaimana dengan video game di mana Anda melihat ribuan piksel? Anda tidak bisa menulis resep sederhana untuk itu!

Kita membutuhkan sesuatu yang dapat melihat situasi yang rumit dan mencari tahu tindakan terbaik. Sesuatu itu adalah **jaringan saraf (neural network)**.

---

## Apa itu Jaringan Saraf?

Pikirkan otak Anda. Jutaan sel kecil yang disebut neuron berbicara satu sama lain. Saat Anda menyentuh sesuatu yang panas, neuron mengirimkan sinyal: "PANAS! → Tarik tangan SEKARANG!" Setiap neuron meneruskan informasi, dan bersama-sama mereka membuat keputusan yang cerdas.

Sebuah **jaringan saraf pada komputer** bekerja dengan cara yang sama:

```
Lapisan Input      Lapisan Tersembunyi 1    Lapisan Tersembunyi 2    Lapisan Output
[pos kereta]  →    [128 neuron]  →          [128 neuron]  →          [skor dorong KIRI]
[kecepatan krt] →   [  ...       ]          [  ...       ]          [skor dorong KANAN]
[sudut tiang] →
[kecepatan tiang] →
```

Setiap panah memiliki **bobot (weight)** (seberapa kuat koneksi tersebut). Ada ribuan bobot ini — dan jaringan mempelajari SEMUANYA!

**Contoh kehidupan nyata:** Seorang koki di restoran mencicipi makanan Anda dan menyesuaikan ratusan bahan sekaligus. Setiap kuncup pengecap seperti neuron, dan bersama-sama mereka memberi tahu koki "tambahkan lebih banyak garam" atau "kurangi lada." Melatih jaringan itu seperti koki yang belajar melalui ribuan hidangan.

---

## DQN = Deep Q-Network

**DQN** (Deep Q-Network) ditemukan oleh DeepMind pada tahun 2013. Mereka mengambil formula Q-learning lama dan menukar tabel-Q dengan jaringan saraf!

Alih-alih:
> Tabel-Q[status][tindakan] = skor

Kita memiliki:
> Jaringan-Q(status) → [skor_untuk_kiri, skor_untuk_kanan]

Jaringan mengambil status sebagai input dan mengeluarkan nilai-Q untuk SEMUA tindakan sekaligus. Ini jauh lebih efisien daripada menghitungnya secara terpisah!

---

## Skrip Ini: Versi "Naif"

Skrip ini menunjukkan DQN **tanpa** trik khusus apa pun. Ia hanya:
1. Melihat status
2. Bertanya pada jaringan "seberapa baik kiri? seberapa baik kanan?"
3. Melakukan tindakan dengan skor yang lebih tinggi
4. Mendapatkan imbalan, memperbarui jaringan

**Ini sengaja dibuat tidak stabil!** Bayangkan seperti seorang siswa yang langsung melupakan pelajaran sebelumnya setiap kali mereka mempelajari sesuatu yang baru. Jaringan diperbarui setelah setiap langkah, yang menyebabkan kekacauan.

**Contoh kehidupan nyata:** Bayangkan belajar memasak dengan mengubah seluruh resep Anda setelah setiap gigitan. Anda mungkin berpindah dari "terlalu asin" ke "tidak ada garam sama sekali" ke "terlalu asin lagi" dan tidak pernah menetap pada jumlah yang tepat. Itulah yang terjadi di sini!

---

## Apa yang Akan Anda Lihat

Saat Anda menjalankan `dqn_cartpole.py`:
- Skor mungkin melonjak tak menentu (pembelajaran tidak stabil)
- Terkadang agen menjadi sangat mahir, lalu melupakan segalanya
- Plot loss menunjukkan ayunan yang liar

**Ini sudah diperkirakan!** Ini menunjukkan MENGAPA kita membutuhkan peningkatan — experience replay dan target network. Itu akan dibahas selanjutnya!

---

## Trik ε-Greedy 🎲

Robot tidak selalu memilih tindakan terbaik. Terkadang ia memilih secara acak!

Mengapa? Karena jika ia selalu memilih apa yang tampaknya terbaik, ia mungkin tidak akan pernah menemukan pilihan yang lebih baik.

> Dengan probabilitas ε (epsilon): pilih tindakan ACAK (eksplorasi!)
> Dengan probabilitas 1-ε: pilih tindakan TERBAIK yang diketahui (eksploitasi!)

Kita mulai dengan ε = 1.0 (100% acak) dan perlahan-lahan berkurang menjadi ε = 0.01 (1% acak). Dengan cara ini, robot bereksplorasi banyak di awal, lalu fokus pada apa yang telah dipelajarinya.

**Contoh kehidupan nyata:** Saat mengunjungi kota baru, Anda mungkin mencoba restoran acak di awal (eksplorasi). Setelah beberapa lama, Anda kembali ke favorit Anda (eksploitasi). Tapi Anda masih sesekali mencoba sesuatu yang baru untuk berjaga-jaga jika ada permata tersembunyi!

---

## Kosakata Kunci

| Kata | Makna |
|------|---------|
| **Neural Network** | Lapisan-lapisan neuron matematika terhubung yang belajar dari data |
| **Deep** | Lebih dari satu lapisan tersembunyi (karenanya disebut "deep" learning) |
| **DQN** | Deep Q-Network — menggunakan jaringan saraf alih-alih tabel-Q |
| **ε-Greedy** | Strategi: bereksplorasi secara acak kadang-kadang, mengeksploitasi pengetahuan terbaik di lain waktu |
| **Instability** | Jaringan terus "lupa" karena pembaruan saling mengganggu satu sama lain |

---

## Apa yang Hilang (dan Mengapa Itu Penting)

DQN naif ini memiliki dua masalah besar:

1. **Pembaruan yang berkorelasi**: Setiap pengalaman datang secara berurutan (langkah 1, langkah 2, langkah 3...). Jika langkah 5 buruk, SEMUA pembaruan terdekat menjadi bingung bersama-sama.
   
2. **Target yang bergerak**: Setelah setiap pembaruan, jaringan berubah. Tetapi pembaruan berikutnya menggunakan jaringan yang SAMA untuk menghitung targetnya. Ini seperti menembak ke arah sasaran yang bergerak!

Masalah-masalah ini diselesaikan oleh **Experience Replay** dan **Target Networks** di skrip berikutnya. Bersama-sama, mereka mengubah DQN dari pemula yang goyah menjadi juara pemain game!
