# SARSA untuk Cliff Walking 🏔️

## Apa Itu?

Bayangkan sebuah **lorong yang sangat panjang** dengan **tebing yang mengerikan** di sepanjang salah satu sisinya. Jika Anda jatuh dari tebing, Anda harus kembali ke titik awal! Tujuan Anda adalah berjalan dari satu ujung ke ujung lainnya secepat mungkin, tanpa jatuh.

**SARSA** adalah robot yang belajar menyusuri lorong ini dengan berlatih. Ia belajar untuk mengambil **jalur aman** yang menghindari tebing — meskipun sedikit lebih panjang — karena ia tahu ia mungkin secara tidak sengaja tergelincir ke dekat tepi tebing saat bereksplorasi!

---

## Ide Besar: Belajar dari Apa yang Benar-benar Anda Lakukan

SARSA adalah singkatan dari: **S**tate → **A**ction → **R**eward → **S**tate → **A**ction (Status → Tindakan → Imbalan → Status → Tindakan)

Ini adalah lima keping informasi yang digunakan SARSA untuk belajar:

1. **S** — Di mana saya sekarang? (status saat ini)
2. **A** — Tindakan apa yang benar-benar saya ambil?
3. **R** — Imbalan apa yang saya dapatkan?
4. **S** — Di mana saya berakhir?
5. **A** — Tindakan apa yang *benar-benar akan saya ambil selanjutnya*?

Huruf "A" yang terakhir inilah yang membuat SARSA istimewa! Ia melakukan pembaruan menggunakan tindakan yang *benar-benar akan ia ambil berikutnya* (bahkan jika itu adalah gerakan eksplorasi acak), bukan tindakan ideal yang sempurna.

**Contoh kehidupan nyata:** Pikirkan tentang belajar naik sepeda. Jika Anda tahu Anda terkadang goyah secara acak (eksplorasi), Anda akan menjauh sedikit dari mobil yang diparkir — karena Anda tahu diri Anda yang goyah mungkin akan meliuk! SARSA melakukan ini: ia mempelajari jalur yang aman karena ia memperhitungkan kesalahan acaknya sendiri.

---

## Peta Cliff Walking

```
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[S][C][C][C][C][C][C][C][C][C][C][G]
   ← ← ← ← TEBING ← ← ← ← ←
```

- **S** = Start (kiri bawah)
- **G** = Goal (kanan bawah)
- **C** = Cliff (Tebing) — melangkah ke sini = -100 imbalan, mulai ulang!
- Setiap langkah lainnya = -1 imbalan

---

## Apa yang Ditemukan Kode Kami

Setelah melatih SARSA selama 500 episode:

| Hasil | Nilai |
|--------|-------|
| Rata-rata imbalan 50-episode terakhir | **-21,6** |
| Imbalan jalur optimal (berisiko) | -13 |

Kebijakan yang dipelajari SARSA berjalan **di sepanjang bagian atas kisi** — jalan memutar yang aman! Ini memakan biaya beberapa langkah ekstra (-21 alih-alih -13), tetapi hampir tidak pernah jatuh dari tebing selama pelatihan.

---

## Contoh Kehidupan Nyata

- **Perawat yang memberikan obat**: Mengikuti protokol aman yang terbukti (jalur aman) bahkan jika ada metode yang sedikit lebih cepat, karena kesalahan kecil (eksplorasi) bisa berbahaya.
- **Pilot maskapai penerbangan**: Mengikuti daftar periksa yang ketat (jalur aman) bahkan ketika jalan pintas mungkin tampak lebih cepat, untuk memperhitungkan kesalahan manusia.
- **Belajar memasak**: Mulai dengan resep yang telah teruji (aman), bukan jalan pintas yang berisiko.

---

## Kata Kunci untuk Diingat

- **On-policy**: Belajar tentang kebijakan yang sebenarnya digunakannya (termasuk kesalahan acaknya)
- **Pembaruan SARSA**: Menggunakan tindakan selanjutnya yang *aktual*, bukan tindakan yang terbaik secara teoritis
- **Jalur aman**: Jalur lebih panjang yang menghindari bahaya, memperhitungkan kesalahan eksplorasi
- **Kontrol TD (Temporal Difference)**: Memperbarui nilai setelah setiap langkah tunggal (tidak menunggu seluruh episode selesai)

Ide besarnya: **SARSA jujur — ia belajar dari apa yang benar-benar ia lakukan, bukan apa yang ia harap akan ia lakukan. Ini membuatnya berhati-hati dan aman di dekat bahaya!**
