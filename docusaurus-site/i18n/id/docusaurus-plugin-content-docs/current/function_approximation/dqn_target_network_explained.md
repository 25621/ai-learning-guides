# Target Network: Menstabilkan Titik Bidik 🎯

## Masalah Tiang Gawang yang Bergerak

Bayangkan Anda mencoba mengenai titik tengah sasaran (bullseye) dengan busur dan anak panah. Anda memanah, melihat di mana anak panah mendarat, dan menyesuaikan bidikan Anda untuk waktu berikutnya. Sederhana, bukan?

Sekarang bayangkan titik tengah sasarannya BERGERAK setiap kali Anda memanah! Setiap anak panah yang Anda lepaskan sedikit mengubah di mana sasaran itu akan berada untuk tembakan berikutnya. Anda tidak akan pernah mahir — Anda akan mengejar target yang selalu lari.

Itulah masalah utama pada DQN tanpa target network!

---

## Mengapa Target-Q Terus Bergerak

Dalam DQN, target untuk setiap pembaruan adalah:
> target = imbalan + γ × max(Q(status_berikutnya))

Di sini **γ (gamma)** adalah **faktor diskon (discount factor)** — angka antara 0 dan 1 (biasanya 0,99) yang mengontrol seberapa besar agen peduli pada imbalan *masa depan* vs. imbalan *segera*.

**Contoh kehidupan nyata:** Bayangkan seseorang menawarkan Anda satu kue sekarang, atau dua kue besok. Jika Anda benar-benar ingin kue sekarang, γ Anda rendah (Anda sangat mendiskon masa depan). Jika Anda sabar dan senang menunggu, γ Anda tinggi (imbalan masa depan hampir sama pentingnya dengan sekarang). Dalam RL, γ = 0,99 berarti "imbalan di langkah berikutnya bernilai 99% dari imbalan saat ini."

Nilai-Q di sisi kanan berasal dari... jaringan yang sama yang kita latih!

Jadi setiap kali kita memperbarui jaringan (untuk membuat nilai-Q lebih baik), kita juga mengubah targetnya. Ini adalah lingkaran umpan balik (feedback loop):

1. Perbarui jaringan → nilai-Q berubah
2. Nilai-Q berubah → target berubah
3. Target berubah → perbarui jaringan secara berbeda
4. Ulangi selamanya — tidak stabil!

**Contoh kehidupan nyata:** Mencoba menimbang berat badan Anda pada timbangan yang mengubah pembacaannya setiap kali Anda menginjaknya. Anda tidak akan pernah tahu berat badan asli Anda!

---

## Solusi: Bekukan Sasaran! ❄️

**Target Network** adalah SALINAN dari jaringan-Q utama yang dibekukan di tempatnya.

- **Online network** (`qnet`): Diperbarui di setiap langkah pelatihan — belajar dengan cepat
- **Target network** (`target_net`): Salinan beku — hanya diperbarui setiap 100 langkah

Kita menggunakan target yang BEKU untuk menghitung target:
> target = imbalan + γ × max(Q_TARGET(status_berikutnya))

Target tersebut tetap diam selama 100 langkah! Itu memberi jaringan online tujuan yang stabil untuk dibidik. Kemudian kita menyalin bobot online ke dalam target, bekukan lagi, dan ulangi.

**Contoh kehidupan nyata:** Bayangkan seorang siswa dan seorang guru. Guru memberikan pekerjaan rumah (target). Siswa belajar dan berkembang. Setelah 100 pelajaran, guru MEMPERBARUI pekerjaan rumahnya menjadi lebih sulit. Guru tidak berubah setiap menit — itu akan terlalu kacau!

---

## Resep Lengkap DQN 🍕

Algoritma DQN yang lengkap (experience replay + target network) adalah:

```
1. Inisialisasi jaringan online Q dan jaringan target Q_target (bobot yang sama)
2. Buat replay buffer (kotak memori)

Setiap langkah lingkungan:
  a. Pilih tindakan menggunakan ε-greedy dengan Q
  b. Simpan (status, tindakan, imbalan, status_berikutnya) di buffer

Setiap 4 langkah:
  c. Ambil sampel mini-batch acak dari buffer
  d. Hitung target menggunakan Q_TARGET (beku!)
  e. Perbarui Q untuk meminimalkan loss

Setiap 100 langkah:
  f. Salin bobot Q → Q_TARGET (singkronisasi target)
```

Ini adalah algoritma persis dari makalah DeepMind DQN (2015)!

---

## Apa yang Ditunjukkan Perbandingannya

Saat Anda menjalankan `dqn_target_network.py`, Anda akan melihat:

**Tanpa target network (hanya DQN + replay):**
- Pelatihan mungkin "oke" tetapi dengan kegagalan berkala
- Nilai-Q dapat menyimpang (meledak atau berosilasi)
- Pembelajaran kurang dapat diprediksi

**DQN Lengkap (replay + target network):**
- Pembelajaran yang lebih konsisten meningkat
- Nilai-Q tetap dalam kisaran yang wajar
- Konvergensi lebih cepat ke ambang penyelesaian (195+ di CartPole)

---

## "Tiga Serangkai Maut" (The Deadly Triad) ☠️

Dalam pembelajaran penguatan, menggabungkan tiga hal ini menciptakan ketidakstabilan:

1. **Aproksimasi fungsi (Function approximation)** (jaringan saraf, bukan tabel) ← kita menggunakan ini
2. **Bootstrapping** (menggunakan nilai-Q untuk memperkirakan nilai-Q) ← kita menggunakan ini
3. **Pembelajaran di luar kebijakan (Off-policy learning)** (Q-learning menggunakan max, bukan kebijakan yang sebenarnya) ← kita menggunakan ini

Ketiganya bersama-sama = "tiga serangkai maut." DQN menaklukkan ini dengan:
- Experience replay → memutus korelasi
- Target network → memutus lingkaran umpan balik

Ini tidak sepenuhnya menyelesaikan masalah, tetapi membuatnya dapat dikelola!

---

## Kosakata Kunci

| Kata | Makna |
|------|---------|
| **Target Network** | Salinan beku dari jaringan-Q yang digunakan hanya untuk menghitung target |
| **Online Network** | Jaringan-Q yang sedang dilatih secara aktif |
| **Singkronisasi (Sync)** | Menyalin bobot jaringan online ke dalam jaringan target |
| **Feedback Loop** | Ketika output dari suatu sistem berbalik untuk mengubah input (dapat menyebabkan ketidakstabilan) |
| **Deadly Triad** | Kombinasi aproksimasi fungsi + bootstrapping + off-policy yang menyebabkan ketidakstabilan |

---

## Dampak di Dunia Nyata

Pada tahun 2015, DeepMind menerbitkan makalah DQN mereka yang menunjukkan AI yang dapat memainkan 49 game Atari pada tingkat manusia super — hanya menggunakan dua trik ini (replay + target network).

Sebelum ini, orang mengira Anda tidak bisa melatih jaringan saraf dengan RL karena ketidakstabilan. DeepMind membuktikan mereka salah, dan itu memulai revolusi deep RL!

Selanjutnya, kita akan menerapkan resep lengkap DQN ini pada Atari Pong — video game sungguhan dengan piksel mentah sebagai input!
