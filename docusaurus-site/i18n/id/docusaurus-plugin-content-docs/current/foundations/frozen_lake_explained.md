# Frozen Lake dengan Kebijakan Acak 🧊

## Apa Itu Frozen Lake?

Bayangkan Anda sedang bermain di **kolam beku** bersama teman-teman Anda.

Es tersebut sebagian besar aman, tetapi beberapa titik memiliki **lubang (hole)** — jika Anda menginjak lubang, Anda akan jatuh dan permainan berakhir! Di salah satu ujung kolam ada sebuah **hadiah** 🎁. Tugas Anda adalah meluncur dari titik **awal** ke **hadiah** tanpa jatuh ke lubang.

Berikut tampilan kolam beku tersebut (4 kotak × 4 kotak):

```
S  F  F  F
F  H  F  H
F  F  F  H
H  F  F  G
```

- **S** = Start (titik mulai Anda)
- **F** = Frozen ice (es beku, aman!)
- **H** = Hole (lubang, kalau jatuh berarti game over 😨)
- **G** = Goal — hadiahnya! 🎁

---

## Bagian yang Sulit: Es yang Licin!

Di kolam beku sungguhan, saat Anda mencoba berjalan ke *kanan*, terkadang es membuat Anda meluncur ke *atas* atau ke *bawah*! Itulah yang membuatnya sulit.

Bahkan jika Anda *ingin* pergi ke kanan, permainan mungkin meluncurkan Anda ke tempat lain. Ini disebut **stokastisitas** — kata keren untuk "hal-hal tidak selalu berjalan sesuai rencana Anda."

---

## Apa Itu Kebijakan Acak?

Sebuah **kebijakan (policy)** hanyalah sebuah rencana: "Dalam situasi ini, saya akan melakukan tindakan INI."

**Kebijakan acak** berarti: "Saya tidak punya rencana sama sekali! Saya hanya akan memilih arah acak setiap saat — atas, bawah, kiri, atau kanan — seperti memutar roda undian!"

Ini seperti bayi yang berjalan di atas es tanpa tahu di mana letak hadiahnya.

---

## Apa yang Ditemukan Kode Kami

Kami mencoba kebijakan acak selama **1.000 permainan**:

| Hasil | Nilai |
|--------|-------|
| **Berapa kali mencapai hadiah** | 11 dari 1.000 (1,1%) |
| **Langkah rata-rata per permainan** | 7,5 langkah |
| **Permainan tercepat** | 2 langkah |
| **Permainan terlama** | 33 langkah |

Seringkali, pejalan acak tersebut jatuh ke lubang dengan cepat. Hanya 1 dari 100 permainan yang berakhir dengan menemukan hadiah!

---

## Mengapa Ini Berguna?

Meskipun kebijakan acak itu buruk, ia memberi kita **baseline** — titik awal untuk dibandingkan.

Saat kita nanti membangun kebijakan yang *pintar* (menggunakan Q-learning atau algoritma lainnya), kita bisa berkata: "Agen pintar kita berhasil 75% — jauh lebih baik daripada si pejalan acak yang cuma 1%!"

**Contoh kehidupan nyata:** Bayangkan mencoba mencari ruang kelas Anda di sekolah baru dengan berbelok ke kiri atau ke kanan secara acak di setiap lorong. Anda mungkin sampai di sana suatu saat nanti, tetapi itu akan memakan waktu lama! Kebijakan pintar itu seperti memiliki peta.

---

## Apa yang Ditunjukkan Heatmap

Dalam gambar kami, **heatmap** menunjukkan kotak mana yang paling sering dikunjungi oleh pejalan acak:

- Kotak **Start** sangat sering dikunjungi (setiap permainan dimulai di sana).
- Kotak di dekat **lubang** lebih jarang dikunjungi (si pejalan sering jatuh sebelum mencapainya).
- Kotak **Goal** sangat jarang dikunjungi karena si pejalan acak hampir tidak pernah sampai ke sana.

Ini memberi tahu kita sesuatu yang penting: kebijakan acak terjebak di dekat awal dan tidak pernah benar-benar menjelajahi seluruh kolam beku.

---

## Kata Kunci untuk Diingat

- **Policy**: Rencana Anda tentang apa yang harus dilakukan di setiap situasi
- **Kebijakan acak**: Tidak ada rencana — pilih saja tindakan acak!
- **Baseline**: Hasil buruk yang kita gunakan untuk perbandingan (seberapa lebih baik yang bisa kita lakukan?)
- **Stochastic**: Hal-hal tidak selalu berjalan sesuai rencana (seperti es yang licin!)
- **Success rate**: Seberapa sering kita menang? (Di sini: 1,1% — sangat rendah!)

Ide besarnya: **Kebijakan acak adalah titik awal. Pembelajaran yang sebenarnya berarti membangun rencana yang lebih baik!**
