# Agen Q-Learning untuk Frozen Lake 🧊

## Apa Itu?

Bayangkan sebuah kolam beku dengan es yang licin. Ada kotak **Mulai (Start)** dan kotak **Tujuan (Goal)** dengan beberapa **Lubang (Hole)** di tengahnya. Jika Anda jatuh ke dalam lubang, Anda harus mulai dari awal!

Es tersebut licin, jadi meskipun Anda mencoba berjalan ke kanan, Anda mungkin malah meluncur ke atas atau ke bawah. **Agen Q-Learning** adalah robot yang belajar — dengan mencoba berulang kali — cara berpindah dari Mulai ke Tujuan tanpa jatuh ke dalam lubang!

---

## Apa Kepanjangan "Q" dalam Q-Learning?

**"Q"** adalah singkatan dari **"Quality" (Kualitas)** — khususnya, *kualitas* dari mengambil tindakan tertentu dalam situasi tertentu.

Pikirkan seperti peringkat restoran: "Seberapa baik (kualitasnya) memesan pizza di restoran INI?" Q(s, a) bertanya: "Seberapa baik mengambil tindakan **a** saat saya berada di status **s**?"

Nilai Q yang tinggi berarti: "Pilihan bagus! Tindakan ini menghasilkan banyak imbalan."
Nilai Q yang rendah berarti: "Ide buruk! Tindakan ini biasanya menimbulkan masalah."

**Contoh kehidupan nyata:** Bayangkan Anda adalah seorang anak yang memutuskan apakah akan makan permen sebelum makan malam. Nilai Q Anda untuk "makan permen sekarang" mungkin tinggi saat ini (rasanya enak!) tetapi rendah secara keseluruhan (ibu marah, Anda merasa mual nanti). Q-learning belajar untuk memperhitungkan konsekuensi masa depan tersebut — bukan hanya perasaan saat ini!

---

## Ide Besar: Tabel Skor Ajaib

Q-Learning membangun tabel besar yang disebut **tabel-Q (Q-table)**. Setiap baris adalah kotak di atas es, dan setiap kolom adalah tindakan (kiri, kanan, atas, bawah). Angka-angka di dalamnya adalah **skor**: "Seberapa baik mengambil tindakan ini dari kotak ini?"

Setiap kali robot mencoba sebuah gerakan:
1. Ia mendapatkan umpan balik (apakah ia jatuh? apakah ia sampai di tujuan?)
2. Ia memperbarui skor di tabel menggunakan rumus ini:

> **Skor Baru = Skor Lama + Tingkat Pembelajaran × (Apa yang sebenarnya terjadi − Apa yang saya harapkan)**

Robot pada dasarnya bertanya: "Apakah gerakan ini lebih baik atau lebih buruk daripada yang saya kira?"

**Contoh kehidupan nyata:** Pikirkan seorang bayi yang sedang belajar berjalan. Setiap kali mereka mencoba melangkah dan jatuh, mereka belajar "langkah itu buruk." Setiap kali mereka berhasil, mereka ingat "itu berhasil!" Setelah banyak mencoba, mereka mencari tahu cara berjalan. Q-learning melakukan hal yang sama, tetapi dengan tabel!

---

## Apa yang Membuat Q-Learning Spesial: Ia "Off-Policy"!

Inilah sesuatu yang cerdas: ketika Q-Learning memperbarui tabelnya, ia *selalu mengasumsikan ia akan melakukan gerakan sempurna di lain waktu*, meskipun selama pelatihan ia terkadang mencoba gerakan acak.

Ini membuat Q-Learning menjadi **off-policy**: strategi yang ia *pelajari* (selalu pilih tindakan terbaik yang diketahui) terpisah dari strategi yang ia *ikuti* selama pelatihan (terkadang pilih tindakan acak untuk bereksplorasi). Secara konkret, pembaruan tabel-Q menggunakan nilai-Q *maksimum* dari status berikutnya — yang terbaik secara teoritis — bahkan ketika langkah robot berikutnya yang sebenarnya mungkin acak.

Dalam bahasa sederhana: robot mungkin berkeliaran secara acak ke kiri untuk bereksplorasi, tetapi pembelajarannya tetap menghitung seolah-olah ia akan mengambil tindakan *terbaik* berikutnya. Pemisahan ini membiarkan Q-Learning konvergen ke strategi optimal tidak peduli seberapa banyak ia bereksplorasi.

---

## Apa yang Ditemukan Kode Kami

Kami melatih selama **50.000 episode** pada Frozen Lake 4×4 yang licin:

| Metrik | Hasil |
|--------|--------|
| Tingkat keberhasilan evaluasi rakus (greedy) | **73,1%** |
| Target pencapaian (>70%) | ✓ **BERHASIL** |

Esnya sangat licin, jadi kebijakan terbaik pun tidak bisa menang 100% setiap saat!

Tabel-Q yang dipelajari menunjukkan agen berhasil menemukan cara: pergi ke bawah dan ke kanan sambil menghindari lubang.

---

## Contoh Kehidupan Nyata

- **Mobil otonom**: Mempelajari lajur mana yang harus diambil di persimpangan melalui uji coba.
- **Sistem rekomendasi**: Mempelajari film mana yang akan disarankan berdasarkan apakah pengguna menyukai saran sebelumnya.
- **AI video game**: Karakter yang belajar menavigasi labirin dengan mencoba banyak jalur.

---

## Kata Kunci untuk Diingat

- **Q-table**: Tabel berisi "seberapa baik setiap tindakan di setiap status"
- **Q(s, a)**: Skor untuk mengambil tindakan a di status s
- **Reward (Imbalan)**: Apa yang didapat agen setelah mengambil tindakan (+1 untuk mencapai tujuan, 0 sebaliknya)
- **Off-policy**: Mempelajari strategi optimal bahkan saat menjelajah secara acak
- **ε-greedy** (ε = epsilon): Sebagian besar waktu melakukan tindakan terbaik yang diketahui; terkadang menjelajah secara acak
- **Faktor diskon γ** (γ = gamma): Seberapa berharga imbalan masa depan (seperti lebih suka uang sekarang daripada nanti)

Ide besarnya: **Q-Learning membangun "lembar contekan" untuk setiap situasi, dan terus memperbaikinya sampai ia tahu gerakan terbaik di mana saja.**
