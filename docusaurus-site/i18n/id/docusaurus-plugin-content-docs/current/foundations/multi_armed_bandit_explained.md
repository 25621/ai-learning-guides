# Masalah Multi-Armed Bandit 🎰

## Apa Itu?

Bayangkan Anda berada di pesta ulang tahun dan ada **10 toples permen yang berbeda**. Setiap toples memiliki permen di dalamnya, tetapi beberapa toples memiliki permen yang *enak* dan beberapa toples memiliki permen yang *tidak terlalu enak*. Anda tidak tahu toples mana yang terbaik — Anda harus mencobanya!

Setiap kali Anda mengambil permen dari toples, Anda mendapatkan beberapa permen. Tugas Anda adalah:

> **Dapatkan permen enak sebanyak mungkin!**

Itulah masalah Multi-Armed Bandit! Alih-alih toples permen, para ilmuwan menyebutnya "lengan/arm" (seperti lengan pada mesin slot). Setiap lengan memberi Anda hadiah, tetapi hadiahnya berbeda-beda setiap saat.

---

## Pertanyaan Besar: Haruskah Saya Mencoba Toples Baru atau Tetap dengan Favorit Saya?

Ini adalah bagian tersulit! Katakanlah Anda mencoba Toples #3 dan rasanya cukup enak. Sekarang Anda punya pilihan:

- **Eksploitasi (Exploit)**: Terus pilih Toples #3 karena Anda sudah tahu itu enak.
- **Eksplorasi (Explore)**: Coba toples baru — mungkin Toples #7 bahkan *lebih enak*!

Jika Anda hanya memilih toples pertama yang Anda sukai, Anda mungkin melewatkan toples yang sangat enak. Tetapi jika Anda *selalu* mencoba toples baru, Anda tidak pernah menggunakan apa yang sudah Anda pelajari!

**Contoh kehidupan nyata:** Pikirkan tentang restoran favorit Anda. Anda selalu memesan nugget ayam (eksploitasi!), tetapi mungkin pizzanya bahkan lebih enak. Jika Anda tidak pernah mencoba sesuatu yang baru, Anda tidak akan pernah tahu!

---

## Strategi Epsilon-Greedy {#the-epsilon-greedy-strategy}

Cara cerdas untuk menyelesaikan masalah ini disebut **epsilon-greedy** (epsilon hanyalah huruf Yunani ε, diucapkan seperti "ep-sih-lon"):

1. **Sebagian besar waktu (misalnya 90%)**: Pilih toples yang Anda *pikir* adalah yang terbaik.
2. **Kadang-kadang (misalnya 10%)**: Pilih toples *acak* untuk bereksplorasi!

Perjalanan eksplorasi 10% membantu Anda menemukan toples yang lebih baik. Perjalanan eksploitasi 90% membiarkan Anda menggunakan apa yang sudah Anda pelajari.

---

## Apa yang Ditemukan Kode Kami

Kami menguji 10 lengan (toples permen) dengan 200 anak yang berbeda, masing-masing 1000 pilihan:

| Strategi | % Waktu Memilih Toples Terbaik |
|----------|----------------------------------|
| **Tidak pernah eksplorasi (ε=0)** | 14,5% — terjebak lebih awal, tidak pernah menemukan yang terbaik! |
| **Eksplorasi 1% dari waktu (ε=0,01)** | 37,6% — perlahan menemukan toples terbaik |
| **Eksplorasi 10% dari waktu (ε=0,10)** | **74,2%** — belajar dengan cepat, memilih yang terbaik sebagian besar waktu! |

**Pelajaran**: Sedikit eksplorasi sangat bermanfaat!

---

## Contoh Kehidupan Nyata

- **Rekomendasi Netflix**: Haruskah Netflix menampilkan film yang mungkin Anda sukai (eksploitasi) atau menyarankan sesuatu yang baru (eksplorasi)?
- **Dokter memilih pengobatan**: Gunakan pengobatan yang biasanya berhasil (eksploitasi) atau coba yang baru yang mungkin lebih baik (eksplorasi)?
- **Lebah mencari bunga**: Haruskah ia terus mengunjungi bunga yang ia tahu memiliki nektar, atau terbang ke ladang baru?

---

## Kata Kunci untuk Diingat

- **Arm**: Salah satu pilihan (seperti toples permen)
- **Reward**: Apa yang Anda dapatkan saat memilih lengan (seperti permen)
- **Exploit**: Gunakan apa yang sudah Anda ketahui bagus
- **Explore**: Coba sesuatu yang baru untuk belajar lebih banyak
- **Epsilon (ε)**: Peluang Anda bereksplorasi alih-alih mengeksploitasi

Ide besarnya: **Anda harus menyeimbangkan antara mencoba hal-hal baru dengan menggunakan apa yang Anda ketahui!**
