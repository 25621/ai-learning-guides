# SARSA vs Q-Learning: Jalur Aman vs Jalur Optimal 🐢 vs 🐇

## Apa Itu?

Dua robot sama-sama perlu berjalan di sepanjang **tepi tebing** untuk mencapai tujuan. Kedua robot masih dalam tahap *belajar* dan terkadang melakukan gerakan acak (ups!).

- 🐢 **Robot SARSA**: "Saya tahu saya terkadang goyah... jadi saya akan berjalan jauh dari tebing agar aman, meskipun butuh waktu lebih lama."
- 🐇 **Robot Q-Learning**: "Jalur terpendek ada tepat di tepi tebing — ayo berangkat! (Terkadang jatuh saat belajar, tetapi akhirnya mempelajari rute terbaik.)"

Kedua robot itu pintar, tetapi mereka melakukan **tradeoff yang berbeda**: aman-tapi-lebih-lambat vs optimal-tapi-berisiko-saat-belajar.

---

## Perbedaan Utama: "Tindakan Berikutnya" Apa yang Anda Gunakan?

Saat memperbarui skor setelah setiap langkah, kedua algoritma bertanya:
> "Berapa nilai dari *status berikutnya*?"

| Algoritma | Menggunakan tindakan berikutnya... | On-policy? |
|-----------|------------------------|------------|
| **SARSA** | ...yang akan *benar-benar saya ambil* (mungkin acak!) | Ya |
| **Q-Learning** | ...yang *terbaik secara teoritis* (selalu rakus/greedy) | Tidak |

**Contoh kehidupan nyata:** Dua anak belajar naik sepeda.

- **Anak SARSA**: Tetap dekat dengan rumput karena *mereka tahu* mereka terkadang goyah secara acak. Mereka merencanakan untuk diri mereka yang sebenarnya yang masih goyah.
- **Anak Q-Learning**: Berlatih di tengah jalan karena mereka membayangkan diri mereka di masa depan yang sempurna yang tidak pernah goyah. Mereka terkadang jatuh sekarang, tetapi mempelajari jalur terbaik lebih cepat.

Kedua anak tersebut akhirnya belajar — tetapi selama pelatihan, anak SARSA lebih sedikit jatuhnya!

---

## Apa yang Ditemukan Kode Kami

Kedua algoritma berjalan selama **500 episode** pada Cliff Walking dengan ε=0,1 (ε = epsilon; di sini berarti peluang 10% untuk melakukan gerakan acak):

| Metrik | SARSA | Q-Learning |
|--------|-------|------------|
| Rata-rata imbalan selama pelatihan (50 ep terakhir) | **-19,7** | **-51,0** |
| Evaluasi rakus (tanpa eksplorasi) | -17 | **-13** |

- **Selama pelatihan**: SARSA mendapatkan **imbalan yang jauh lebih baik** karena menghindari tebing (memperhitungkan gerakan acaknya sendiri)
- **Setelah pelatihan** (rakus murni): Q-Learning menemukan **jalur optimal yang lebih pendek** (-13)!

Seiring menyusutnya ε menuju 0, kedua algoritma konvergen ke kebijakan optimal yang sama.

---

## Ringkasan Visual

```
Jalur SARSA (selama pelatihan):      Jalur Q-Learning (rakus, setelah pelatihan):
[ ][→][→][→][→][→][→][→][→][→][→][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[↑][→][→][→][→][→][→][→][→][→][→][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[S][C][C][C][C][C][C][C][C][C][C][G]   [S][→][→][→][→][→][→][→][→][→][→][G]
     (jalan memutar aman, baris atas)         (optimal, mepet tebing)
```

---

## Contoh Kehidupan Nyata

- **Ahli bedah baru vs ahli bedah berpengalaman**: Ahli bedah baru (SARSA) menjauh dari teknik berisiko saat belajar. Ahli bedah berpengalaman (Q-Learning rakus) menggunakan teknik yang paling efisien setelah menguasainya.
- **Mengemudi di kota vs rute jalan tol**: Perencanaan gaya SARSA mengambil jalan perumahan yang lebih aman; Q-Learning menemukan jalan tol yang optimal tetapi sempit.
- **Siswa yang sedang belajar**: Siswa-SARSA tetap pada topik yang dipahami dengan baik selama latihan. Siswa-Q-Learning memaksakan diri ke masalah tersulit (lebih banyak gagal) tetapi mempelajari strategi optimal.

---

## Kata Kunci untuk Diingat

- **On-policy** (SARSA): Belajar tentang apa yang *benar-benar Anda lakukan*, termasuk eksplorasi acak
- **Off-policy** (Q-Learning): Belajar tentang perilaku *terbaik yang mungkin* secara terpisah dari apa yang benar-benar Anda lakukan
- **Safe path (Jalur aman)**: Rute lebih panjang yang menghindari bahaya, digunakan ketika eksplorasi menyebabkan kecelakaan
- **Optimal path (Jalur optimal)**: Rute terpendek/imbalan tertinggi, ditemukan ketika tidak ada eksplorasi yang terjadi
- **Tradeoff eksplorasi-eksploitasi**: Keseimbangan antara mencoba hal-hal baru dengan menggunakan apa yang Anda ketahui

Ide besarnya: **SARSA lebih aman selama pelatihan (on-policy), Q-Learning menemukan jalur optimal lebih cepat (off-policy). Mana yang lebih baik tergantung pada apakah jatuh dari tebing itu masalah besar atau tidak!**
