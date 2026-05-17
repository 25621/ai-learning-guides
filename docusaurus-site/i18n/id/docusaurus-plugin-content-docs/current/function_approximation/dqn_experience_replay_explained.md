# Experience Replay: Mengajari Robot untuk Mengingat 🎒

## Masalah: Lupa (dan Kebingungan)

Ingat bagaimana DQN naif tidak stabil? Alasan terbesarnya adalah **pembelajaran yang berkorelasi (correlated learning)**.

Saat robot memainkan game, ia mengalami hal-hal secara berurutan:
> Langkah 1 → Langkah 2 → Langkah 3 → Langkah 4 → ...

Langkah-langkah ini terhubung! Jika robot miring ke kiri di langkah 10, langkah 11 juga akan miring ke kiri. Mereka tidak independen — mereka bergantung satu sama lain.

Saat kita memperbarui jaringan menggunakan langkah-langkah yang berkorelasi ini, itu seperti mencoba belajar sejarah dengan membaca bab yang sama berulang-ulang. Anda akan menjadi sangat mahir pada satu bab tersebut dan melupakan segalanya yang lain!

**Contoh kehidupan nyata:** Bayangkan belajar untuk ujian dengan hanya melatih pekerjaan rumah kemarin. Anda menjadi luar biasa pada masalah-masalah itu saja, tetapi ujian memiliki pertanyaan yang berbeda! Anda perlu berlatih CAMPURAN dari berbagai masalah yang berbeda.

---

## Solusi: Kotak Memori 📦

**Experience Replay** menambahkan kotak memori besar (disebut **replay buffer**) ke robot.

Alih-alih belajar dari pengalaman terbaru, robot:
1. **Menyimpan** setiap pengalaman dalam kotak memori: (status, tindakan, imbalan, status berikutnya)
2. **Memilih secara acak** segelintir memori dari kotak tersebut
3. **Belajar dari campuran acak itu** alih-alih hanya dari langkah terbaru

```
Langkah Game 1 → [simpan di kotak]
Langkah Game 2 → [simpan di kotak]
Langkah Game 3 → [simpan di kotak]
...
Langkah Game 50 → [simpan di kotak] → pilih 64 memori acak → perbarui jaringan
Langkah Game 51 → [simpan di kotak] → pilih 64 memori acak → perbarui jaringan
```

**Contoh kehidupan nyata:** Pikirkan album foto. Anda tidak belajar tentang hidup Anda hanya dengan melihat foto hari ini. Anda membalik-balik foto LAMA juga — campuran kenangan indah dan momen-momen sulit. Ini membantu Anda memahami pola di seluruh hidup Anda, tidak hanya hari ini.

---

## Mengapa Pengambilan Sampel Acak Membantu

Saat kita memilih memori secara acak, kita memutus korelasi. Robot mungkin belajar dari:
- Memori di mana tiang dalam kondisi sempurna (dari 500 langkah lalu)
- Memori di mana tiang hampir jatuh (dari 20 langkah lalu)
- Memori di mana ia beruntung (dari langkah 3)

Campuran acak ini berarti:
✅ Robot belajar dari berbagai situasi
✅ Setiap memori dapat "diputar ulang" berkali-kali (penggunaan pengalaman yang efisien)
✅ Jaringan tidak hanya terpaku pada peristiwa terbaru (overfit)

---

## Pembelajaran Mini-Batch

Alih-alih memperbarui pada SATU pengalaman sekaligus, kita memperbarui pada **64 pengalaman sekaligus** ("mini-batch"). Ini seperti:
- Cara lama: Baca satu kartu flash, uji diri sendiri
- Cara baru: Baca 64 kartu flash yang berbeda, lalu uji diri Anda pada campurannya

Mini-batch membuat sinyal pembelajaran jauh lebih andal dan tidak terlalu berisik.

---

## Periode Pemanasan (Warmup Period)

Kita tidak langsung belajar! Replay buffer butuh beberapa memori terlebih dahulu. Kita menunggu sampai ada setidaknya **500 pengalaman** di dalam kotak sebelum pelatihan dimulai.

**Contoh kehidupan nyata:** Anda tidak akan mencoba memasak makanan sampai Anda mengumpulkan bahan-bahannya. Periode pemanasan itu seperti berbelanja bahan makanan sebelum memasak!

---

## Apa yang Ditunjukkan Perbandingannya

Saat Anda menjalankan `dqn_experience_replay.py`, Anda akan melihat dua kurva pembelajaran:

| DQN Naif | DQN + Replay |
|-----------|-------------|
| Sangat bergelombang | Lebih halus |
| Sering gagal (lupa segalanya) | Peningkatan yang lebih konsisten |
| Variansi tinggi | Variansi rendah |

Versi replay biasanya:
- Mencapai skor bagus dengan lebih andal
- Tidak sering jatuh dari 500 kembali ke 30
- Menunjukkan kemajuan pembelajaran yang lebih stabil

---

## Replay Buffer dalam Kode

```
ReplayBuffer:
  - kapasitas: 10.000 memori (yang tertua dilupakan saat penuh)
  - push(status, tindakan, imbalan, status_berikutnya, selesai)
  - sample(batch_size=64) → batch acak
```

Bayangkan seperti buku catatan dengan 10.000 baris. Saat penuh, Anda menghapus baris tertua dan menulis yang terbaru. Anda selalu belajar dari halaman acak!

---

## Kosakata Kunci

| Kata | Makna |
|------|---------|
| **Experience Replay** | Menyimpan dan menggunakan kembali pengalaman masa lalu secara acak untuk pelatihan |
| **Replay Buffer** | Kotak memori yang menyimpan tupel (status, tindakan, imbalan, status_berikutnya) masa lalu |
| **Correlated updates** | Saat data pelatihan bergantung pada dirinya sendiri (buruk untuk pembelajaran!) |
| **Mini-batch** | Sampel acak kecil dari memori yang digunakan untuk satu langkah pembaruan |
| **Decorrelation** | Memutus hubungan antara pengalaman berturut-turut |

---

## Apa yang Masih Kurang?

Bahkan dengan replay buffer, ada masalah lain: **target yang bergerak (moving target)**.

Setiap kali kita memperbarui jaringan, nilai-Q berubah. Tetapi nilai-Q yang diperbarui itu JUGA digunakan untuk menghitung target untuk pembaruan BERIKUTNYA. Ini adalah lingkaran kebingungan!

Masalah ini diselesaikan oleh **Target Network** — salinan beku dari jaringan yang hanya diperbarui setiap 100 langkah. Itu membuat "titik bidik" tetap diam untuk sementara waktu sehingga robot dapat membidik dengan andal!
