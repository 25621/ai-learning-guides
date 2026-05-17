# Kontrol Monte Carlo untuk Blackjack 🃏

## Apa Itu?

Pernahkah Anda memainkan permainan kartu di mana Anda harus memutuskan: **"Apakah saya mengambil kartu lagi, atau saya puas dengan apa yang saya miliki?"**

**Blackjack** (juga disebut "21") persis seperti itu! Anda ingin kartu Anda berjumlah sedekat mungkin dengan 21, tanpa melebihinya. Jika Anda melebihi 21, Anda "bust" dan kalah!

**Kontrol Monte Carlo** adalah cara robot belajar bermain Blackjack — dengan memainkan *ribuan permainan penuh* dan mengingat apa yang berhasil dan apa yang tidak.

---

## Ide Besar: Belajar dari Cerita Lengkap

Kata "Monte Carlo" berasal dari kasino terkenal di Monako. Dalam matematika, itu berarti: **gunakan eksperimen acak untuk mempelajari sesuatu**.

Begini cara kerjanya:

1. **Mainkan permainan penuh** (satu episode lengkap) menggunakan strategi apa pun yang Anda miliki
2. **Lihat apa yang terjadi**: Apakah Anda menang? Kalah? Seri?
3. **Bekerja mundur**: Apakah memukul (mengambil kartu) pada angka 17 adalah ide yang bagus? Bagaimana dengan pada angka 14?
4. **Perbarui memori Anda**: Ingat apakah setiap keputusan menyebabkan kemenangan atau kekalahan

Lakukan ini selama **500.000 permainan** dan Anda akan menjadi sangat mahir!

**Contoh kehidupan nyata:** Bayangkan belajar memasak dengan membuat 500.000 makanan. Setiap kali, Anda mengingat persis apa yang Anda lakukan — dan apakah makanannya enak. Setelah mencoba cukup banyak, Anda tahu: "Menambahkan terlalu banyak garam pada langkah ini selalu membuatnya buruk." Monte Carlo bekerja dengan cara yang sama!

---

## Perbedaan Utama dari SARSA dan Q-Learning

SARSA dan Q-Learning memperbarui pengetahuan mereka **setelah setiap langkah** (bahkan di tengah episode). Monte Carlo menunggu sampai **seluruh episode selesai**, lalu menoleh ke belakang pada semuanya.

| Metode | Kapan diperbarui? | Butuh episode lengkap? |
|--------|---------------|------------------------|
| **TD (SARSA, Q-Learning)** | Setelah setiap langkah | Tidak |
| **Monte Carlo** | Setelah setiap episode penuh | Ya |

Ini membuat Monte Carlo lebih sederhana untuk dipahami, tetapi ia tidak bisa belajar sampai setiap episode berakhir.

---

## Status Blackjack

Robot melihat 3 hal setiap giliran:
1. **Total kartu saya** (12 hingga 21)
2. **Kartu apa yang ditunjukkan dealer?** (As hingga 10)
3. **Apakah saya memiliki As yang dapat digunakan (usable Ace)?** (Kartu As dapat bernilai 1 atau 11)

Dari 3 keping informasi ini, ia memutuskan: **Pukul (ambil kartu) atau Diam (berhenti)**?

---

## Apa yang Ditemukan Kode Kami

Setelah **500.000 permainan** Blackjack:

| Hasil | Persentase |
|---------|------------|
| **Menang** | **43,1%** |
| **Seri** | 8,9% |
| **Kalah** | 48,0% |

Ini mendekati "strategi dasar" yang optimal secara matematis (sekitar 42-43% menang)! Robot belajar kapan harus memukul dan kapan harus diam — hanya dengan bermain game dan mengingat.

Kebijakan yang dipelajari menunjukkan:
- **Pukul (Hit)** (ambil kartu) saat total Anda rendah (Anda tidak mungkin bust)
- **Diam (Stick)** saat total Anda tinggi (Anda mungkin bust jika mengambil kartu lain)
- Memiliki **As yang dapat digunakan** memungkinkan Anda menjadi lebih agresif (ia dapat beralih dari 11 ke 1 jika diperlukan)

---

## Contoh Kehidupan Nyata

- **Prakiraan cuaca**: Simulasi Monte Carlo menjalankan ribuan skenario "bagaimana jika" untuk memprediksi cuaca besok.
- **Pemodelan pasar saham**: Analis mensimulasikan ribuan kemungkinan masa depan untuk memperkirakan risiko.
- **Belajar bermain catur**: Seorang pemain meninjau seluruh permainan (bukan hanya langkah tunggal) untuk memahami strategi apa yang menyebabkan kemenangan.

---

## Kata Kunci untuk Diingat

- **Episode**: Satu permainan lengkap dari awal hingga akhir
- **Return (G)**: Total imbalan yang dikumpulkan dari satu titik dalam permainan sampai akhir
- **Every-visit MC**: Memperbarui skor untuk suatu status setiap kali Anda mengunjunginya dalam sebuah episode
- **Tanpa bootstrapping**: Monte Carlo tidak menggunakan estimasi nilai masa depan — ia menunggu hasil nyata!
- **Kebijakan ε-soft** (ε = epsilon): Biasanya melakukan tindakan terbaik yang diketahui, tetapi terkadang bereksplorasi secara acak

Ide besarnya: **Monte Carlo belajar dengan memainkan banyak permainan lengkap. Ini seperti belajar dari pengalaman — Anda mengingat semua yang terjadi dan mencari tahu apa yang menyebabkan kemenangan!**
