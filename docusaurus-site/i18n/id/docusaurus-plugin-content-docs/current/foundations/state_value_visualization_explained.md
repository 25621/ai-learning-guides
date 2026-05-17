# Fungsi Nilai-Status 🗺️

## Apa Itu "Status"?

Bayangkan Anda sedang memainkan permainan papan. Pada setiap saat, Anda berdiri di *satu* kotak di papan tersebut. Kotak itu adalah **status (state)** Anda — di situlah Anda berada saat ini.

Dalam permainan kisi 4×4 kami, ada 16 kotak (status). Setiap kotak adalah tempat yang bisa ditempati oleh agen.

---

## Apa Itu "Nilai"?

Sekarang inilah pertanyaan ajaibnya: **"Jika saya berdiri di kotak ini sekarang, berapa banyak harta karun yang dapat saya harapkan untuk dikumpulkan sebelum permainan berakhir?"**

Jawaban itu adalah **nilai (value)** dari status tersebut!

Kotak dengan **nilai tinggi** berarti: "Ini adalah tempat yang bagus — saya mungkin akan mengumpulkan banyak harta karun dari sini!"

Kotak dengan **nilai rendah** berarti: "Aduh — dari sini, biasanya segalanya berakhir buruk."

**Contoh kehidupan nyata:** Bayangkan Anda sedang bermain petak umpet. Jika Anda bersembunyi di balik pohon besar (tempat yang bagus), peluang Anda untuk menang tinggi — itu adalah status bernilai tinggi! Jika Anda bersembunyi di tengah ruangan kosong, Anda mungkin akan ditemukan — itu adalah status bernilai rendah.

---

## Dunia Kisi (Grid World) Kami

Inilah papan permainan yang kami gunakan:

```
S  .  .  .      S = Start (Mulai)
.  H  .  H      H = Hole (Lubang; imbalan -1, permainan berakhir)
.  .  .  H      G = Goal (Tujuan; imbalan +1, permainan berakhir)
H  .  .  G      . = Kotak kosong yang aman
```

- Jika Anda mencapai **G** (Tujuan): Anda mendapat **+1 poin** 🎉
- Jika Anda menginjak **H** (Lubang): Anda mendapat **-1 poin** 😢
- Langkah lainnya: **0 poin**

Kami menggunakan γ (gamma) = 0,99, yang berarti imbalan masa depan hampir sama berharganya dengan imbalan segera. (Permen besok hampir sama enaknya dengan permen hari ini!)

---

## Dua Rencana (Kebijakan) Berbeda

Kami menguji dua kebijakan dan menghitung nilai setiap kotak untuk masing-masing:

### Kebijakan 1: Acak Seragam
Memilih atas, bawah, kiri, atau kanan secara acak dengan peluang yang sama.

```
Nilai (Kebijakan Acak Seragam):
-0,912  -0,932  -0,912  -0,942
-0,929   (H)   -0,898   (H)
-0,901  -0,801  -0,696   (H)
 (H)   -0,630  -0,104   (G)
```

Hampir semua kotak bernilai **negatif** — kebijakan acak sangat sering jatuh ke lubang sehingga berada di mana pun terasa cukup buruk!

---

### Kebijakan 2: Bias Menuju Tujuan
Lebih suka bergerak ke kanan dan ke bawah (menuju tujuan), tetapi terkadang masih pergi ke arah lain.

```
Nilai (Kebijakan Bias-Menuju-Tujuan):
-0,838  -0,895  -0,814  -0,961
-0,798   (H)   -0,665   (H)
-0,595  -0,143  -0,213   (H)
 (H)    0,254   0,673   (G)
```

Sekarang kotak-kotak di dekat **Tujuan** memiliki **nilai positif** (0,254 dan 0,673)! Kebijakan yang cerdas membuat kotak-kotak tersebut menjadi tempat yang baik untuk dikunjungi.

---

## Arti Warna dalam Gambar Kami

Dalam visualisasi kami:
- **Kotak hijau** = nilai tinggi (tempat yang bagus untuk dikunjungi)
- **Kotak merah** = nilai rendah (hindari tempat ini!)
- **Kotak kuning** = di antaranya

Anda dapat melihat **gradien** — nilai menjadi lebih hijau saat Anda bergerak menuju Tujuan dan lebih merah di dekat Lubang.

---

## Mengapa Kita Peduli dengan Nilai?

Nilai adalah *fondasi* dari pembelajaran penguatan! Begitu Anda mengetahui nilai dari setiap status, Anda dapat membuat keputusan yang cerdas:

> "Saya di kotak A. Saya bisa pergi ke kotak B (nilai = 0,5) atau kotak C (nilai = -0,3). Saya akan pergi ke B — nilainya lebih tinggi!"

Inilah tepatnya cara banyak algoritma RL (seperti Q-learning) belajar membuat keputusan yang baik tanpa diberi tahu aturannya.

**Contoh kehidupan nyata:** Bayangkan Anda sedang memilih antrean mana di supermarket. Setiap antrean adalah sebuah "status." Nilai dari status tersebut adalah seberapa cepat Anda akan menyelesaikan pembayaran. Anda melihat antrean (mengamati status) dan memilih yang memiliki nilai tertinggi (tunggu tersingkat + barang paling sedikit).

---

## Bagaimana Kami Menghitung Nilainya

Kami menggunakan **Evaluasi Kebijakan Iteratif (Iterative Policy Evaluation)**, yang bekerja seperti ini:

1. Mulai: asumsikan semua nilai adalah 0.
2. Perbarui: untuk setiap kotak, hitung nilai yang *seharusnya* berdasarkan ke mana kebijakan membawa Anda selanjutnya.
3. Ulangi sampai nilainya berhenti berubah (konvergen).

Secara matematis: **V(s) = Σ_a π(a|s) × [R(s,a) + γ × V(status_berikutnya)]**

Dalam bahasa sederhana: "Nilai kotak ini = rata-rata imbalan yang akan saya dapatkan sekarang + sedikit dari nilai di mana pun saya berakhir."

---

## Kata Kunci untuk Diingat

- **Status (State)**: Di mana Anda berada saat ini (satu kotak di papan)
- **Nilai V(s)**: Total imbalan yang diharapkan mulai dari status s
- **Kebijakan (Policy)**: Rencana Anda tentang apa yang harus dilakukan di setiap status
- **Faktor diskon γ**: Seberapa besar Anda peduli dengan imbalan masa depan (0,99 = sangat peduli!)
- **Evaluasi Kebijakan**: Menghitung nilai untuk setiap status di bawah kebijakan tertentu

Ide besarnya: **Beberapa tempat lebih baik daripada yang lain — dan fungsi nilai memberi tahu Anda seberapa baik tepatnya setiap tempat itu!**
