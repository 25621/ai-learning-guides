# Dyna-Q: Belajar Lebih Cepat dengan Berimajinasi 🧠

## Apa Itu? {#what-is-it}

Bayangkan seorang anak bernama Mia yang sedang belajar menelusuri sekolah barunya. Setiap hari ia berjalan di lorong-lorong dan menemukan hal-hal baru: "Perpustakaan ada di sebelah kantin," "Ruang Pak Smith ada di lantai atas dekat tangga."

Seorang siswa **Q-learning biasa** hanya belajar dari apa yang ia lakukan *hari ini*. Jika hari ini ia hanya berjalan dari kelas ke kantin, ia hanya memperbarui ingatannya tentang satu jalur itu saja.

Siswa **Dyna-Q** berbeda. Setelah setiap perjalanan nyata, ia duduk sejenak dan **memutar ulang dalam kepalanya** beberapa perjalanan masa lalu yang ia ingat. Setiap putaran ulang memperkuat peta mentalnya. Setelah beberapa minggu ia hafal seluruh sekolah luar-dalam — bukan karena ia lebih banyak berjalan, tetapi karena ia **lebih banyak memikirkan apa yang sudah ia lihat**.

Itulah tepatnya yang dilakukan Dyna-Q untuk agen RL: ia belajar dari pengalaman nyata **dan** dari pengalaman imajiner yang diambil dari model yang ia bangun di sepanjang jalan.

---

## Tiga Bahan Utama {#the-three-ingredients}

Dyna-Q adalah "Q-learning + model + perencanaan". Satu langkah nyata melakukan **tiga** tugas:

1. **Direct RL** — pembaruan Q-learning biasa dari `(s, a, r, s')`.
2. **Pembelajaran model (Model learning)** — mencatat: "Saat saya melakukan *a* di *s*, saya mendapatkan *r* dan berakhir di *s'*."
3. **Perencanaan (Planning)** — mengambil *n* pasangan `(s, a)` acak dari memori model dan melakukan *n* kali pembaruan Q-learning lagi, **berpura-pura** seolah-olah langkah-langkah itu baru saja terjadi.

Langkah ketiga itulah keajaibannya. Dengan `n = 50`, setiap langkah nyata di dunia nyata menyebabkan **51 pembaruan** pada tabel-Q. Agen belajar ~50x lebih cepat — dalam hal langkah nyata — daripada pembelajar Q murni.

---

## Gambar Loop {#a-picture-of-the-loop}

```
                   ┌────────────────────────────────────┐
                   │                                    │
   dunia nyata ──► ambil tindakan a ──► amati (r, s')    │
                            │                           │
              ┌─────────────┼──────────────┐            │
              ▼             ▼              ▼            │
        Q-learning      Model[s,a] ← (r,s')   Perencanaan ─┘
         update                            (n pembaruan imajiner)
```

Modelnya hanyalah tabel pencarian:
`(status, tindakan) → (imbalan, status_berikutnya)`. Murah untuk dibangun, murah untuk ditanyakan.

---

## Contoh Kehidupan Nyata {#real-life-examples}

- **Belajar catur.** Pecatur tingkat Master menghabiskan berjam-jam memutar ulang permainan mereka sendiri dan permainan para ahli di kepala mereka. Setiap putaran ulang adalah "perencanaan" — pembelajaran tambahan dari pengalaman yang sudah terjadi.
- **Musisi yang melatih tangga nada.** Setelah memainkan satu bar yang sulit sekali, mereka melatihnya secara mental sepuluh kali lagi sebelum melanjutkan. Jari-jari tidak bergerak, tetapi otak sedang memperbarui.
- **Mobil otonom.** Saat berhenti di lampu merah, ia memutar ulang seratus kali perpindahan jalur terakhir dalam simulasi untuk menyempurnakan kebijakannya tanpa menghabiskan ban.

---

## Apa yang Dilakukan Kode Kami {#what-our-code-does}

Kami menggunakan **Dyna Maze** klasik ([Sutton & Barto, Gambar 8.2](http://incompleteideas.net/book/the-book.html)): kisi 6×9 dengan beberapa dinding, titik mulai `S` di tengah-kiri, dan tujuan `G` di kanan atas.

Kami menjalankan tiga varian, masing-masing dirata-ratakan di 30 seed acak:

| Pengaturan | Langkah perencanaan per langkah nyata | Makna |
|---------|------------------------------|---------|
| `n = 0` | 0 | Q-learning biasa |
| `n = 5` | 5 | sedikit latihan imajiner |
| `n = 50` | 50 | banyak latihan imajiner |

Skrip ini melaporkan **jumlah rata-rata langkah nyata per episode** seiring berjalannya pelatihan. Semakin sedikit langkah berarti agen telah mempelajari jalur yang lebih langsung ke tujuan.

### Apa yang seharusnya Anda lihat saat menjalankannya {#what-you-should-see-when-you-run-it}

Jalur terpendek di labirin ini adalah ~9 langkah; dengan eksplorasi ε-greedy agen yang terlatih baik rata-rata menempuh ~10 langkah per episode. Jalankan selama 50 episode dan ketiga pengaturan akan konvergen ke sana — perbedaannya adalah *seberapa cepat*:

| Pengaturan | Langkah per episode (10 eps terakhir) | Apa artinya |
|---------|--------------------------------:|---------------|
| `n = 0`   | ~10 | Konvergen — tetapi butuh ~30–50 episode pengembaraan untuk sampai ke sini |
| `n = 5`   | ~10 | Konvergen dalam ~10 episode |
| `n = 50`  | ~10 | Konvergen dalam ~3–5 episode |

Sinyal yang menarik adalah *kurva pembelajaran*, bukan angka akhirnya. Plot yang disimpan di `outputs/dyna_q.png` menunjukkan tiga kurva menukik ke bawah dengan tingkat kecepatan yang sangat berbeda: `n = 50` mencapainya dalam segelintir episode, sementara `n = 0` masih turun di tengah jalan. (Di labirin deterministik kecil seperti ini, Q-learning biasa memang akhirnya sampai ke sana — Dyna-Q hanya membutuhkan jauh lebih sedikit episode nyata, yang merupakan poin utamanya pada lingkungan di mana langkah nyata itu mahal.)

---

## Mengapa Ini Berhasil Sangat Baik di Labirin Ini {#why-it-works-so-well-on-this-maze}

Dua alasan:

1. **Lingkungannya deterministik.** Setiap `(s, a)` selalu memberikan `(r, s')` yang sama, sehingga modelnya tepat setelah satu kunjungan. Pengalaman imajiner sama bagusnya dengan pengalaman nyata.
2. **Langkah nyata itu mahal, langkah imajiner itu gratis.** Setiap pembaruan imajiner hanyalah beberapa pencarian tabel, sementara langkah nyata mengharuskan agen untuk berjalan. Saat interaksi nyata memakan biaya besar (pikirkan: robot nyata, game nyata), Dyna-Q sangat efisien sampel (sample-efficient).

---

## Di Mana Dyna-Q Mengalami Kesulitan {#where-dyna-q-struggles}

- **Lingkungan stokastik.** Jika `(s, a)` dapat mengarah ke banyak nilai `s'` yang berbeda, model "ingat hasil terakhir" akan membohongi Anda. Solusi: simpan jumlah kunjungan atau latih model probabilistik.
- **Lingkungan non-stasioner.** Jika dunianya berubah — misalnya, pintu yang tadinya terbuka tiba-tiba tertutup, atau muncul jalan pintas — model menjadi usang dan memberikan prediksi yang salah. **Dyna-Q+** memperbaiki hal ini dengan menambahkan *bonus eksplorasi*: status yang sudah lama tidak dikunjungi kembali menerima imbalan ekstra kecil, mendorong agen untuk kembali dan memeriksa apakah dunia telah berubah.
- **Ruang status besar.** Kamus dengan kunci `(s, a)` tidak dapat diskalakan ke jutaan status atau ke status kontinu. Itulah celah yang diisi oleh **learned (neural-network) world models** — lihat `world_model.py` berikutnya.

---

## Kata Kunci untuk Diingat {#key-words-to-remember}

| Kata | Makna |
|------|---------|
| **Model**       | Memori tentang `(status, tindakan) → (imbalan, status_berikutnya)` |
| **Planning step** | Melakukan pembaruan-Q menggunakan data imajiner dari model |
| **Direct RL**   | Pembaruan-Q menggunakan data nyata |
| **Efisiensi sampel** | Mengukur seberapa efektif model atau algoritma AI menggunakan data pelatihan untuk mencapai tingkat kinerja tertentu |
| **Dyna**        | Arsitektur Sutton yang menyelingi pembelajaran + perencanaan |

---

## Ringkasan Satu Kalimat {#one-sentence-summary}

> **Dyna-Q belajar dari tindakan nyata DAN dari imajinasi — dan berimajinasi itu gratis.**

Ide ini, dalam bentuk saraf modernnya, mendasari beberapa agen RL terkuat yang pernah dibangun (MuZero, Dreamer, World Models).
