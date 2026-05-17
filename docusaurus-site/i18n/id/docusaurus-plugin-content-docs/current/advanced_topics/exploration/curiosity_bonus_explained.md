# Bonus Keingintahuan (Motivasi Intrinsik) 🧭

## Apa Itu? {#what-is-it}

Bayangkan seorang balita yang ditaruh di ruangan baru. Tidak ada yang membayar mereka, tidak ada yang bertepuk tangan — namun mereka langsung menuju lemari yang belum mereka buka, tombol yang belum mereka tekan, mainan berisik di pojokan. Mereka berjalan berdasarkan **imbalan internal**: *"Itu terlihat baru. Pergi periksa."*

**Bonus keingintahuan (curiosity bonus)** memberikan dorongan internal yang sama kepada agen pembelajaran penguatan. Imbalan nyata dari lingkungan (imbalan "ekstrinsik" — poin, uang, memenangkan permainan) dibiarkan persis seperti apa adanya. Kita cukup menambahkan imbalan kedua yang dihasilkan sendiri karena mengunjungi hal-hal yang menurut agen *baru* atau *mengejutkan*, dan melatih pada jumlah keduanya:

```
imbalan yang dipelajari agen  =  imbalan nyata  +  beta * bonus keingintahuan
```

`beta` adalah tombol yang dimulai besar (jadilah penasaran!) dan menyusut seiring waktu (berhenti membuang waktu, ambil hasil dari apa yang sudah dipelajari).

## Mengapa Harus Repot? Masalah Imbalan Jarang (Sparse-Reward) {#why-bother-the-sparse-reward-problem}

Agen RL normal belajar dari imbalan yang benar-benar mereka terima. Itu bekerja sangat baik ketika imbalan ada di mana-mana ("+1 setiap langkah Anda tetap tegak" di CartPole). Itu berantakan ketika imbalan bersifat **jarang (sparse)** — nol, nol, nol, ... , nol, dan akhirnya +1 setelah urutan tindakan yang sangat spesifik dan panjang.

Contoh nyata dari imbalan jarang:

- **Montezuma's Revenge** (permainan Atari): poin pertama Anda tiba hanya setelah ~100 langkah presisi — turun tangga, hindari tengkorak, naik tangga, ambil kunci. Sampai saat itu skornya adalah nol datar.
- **Kunci kombinasi.** 9.999 kode salah tidak memberi Anda apa-apa; satu kode memberi Anda hadiah.
- **Penemuan obat / eksperimen ilmiah.** Ribuan percobaan gagal, lalu satu yang berhasil.
- **Menulis pembuktian atau program yang panjang.** Tidak ada nilai parsial sampai semuanya selesai.

Agen yang hanya mengandalkan imbalan dalam pengaturan ini seperti siswa yang menolak belajar kecuali mereka dibayar per jawaban benar di ujian akhir — mereka tidak pernah memulai. Keingintahuan adalah bonus yang mengatakan *"menjelajah adalah imbalannya sendiri,"* sehingga agen terus mengorek sampai ia menemukan hadiah yang sebenarnya.

## Dua Jenis Keingintahuan (keduanya diimplementasikan dalam `curiosity_bonus.py`) {#two-flavours-of-curiosity-both-implemented-in-curiosity_bonuspy}

### 1. Kebaruan berbasis hitungan (Count-based novelty): "Saya jarang ke sini" {#1-count-based-novelty-ive-barely-been-here}

Sinyal kebaruan yang paling sederhana. Simpan penghitungan `N(s, a)` tentang berapa kali Anda mengambil tindakan `a` di status `s`, dan berikan diri Anda bonus yang menyusut saat penghitungan itu bertambah:

```
bonus keingintahuan  =  1 / sqrt( N(s, a) + 1 )
```

Pertama kali Anda mencoba sesuatu: bonus = 1.0. Setelah 100 kali percobaan: bonus = 0.1. Setelah 10.000 kali percobaan: 0.01. Agen diberi imbalan karena pergi ke tempat yang belum pernah ia kunjungi, dan daya tarik itu secara alami memudar dari tempat yang sudah sering dikunjungi.

**Analogi kehidupan nyata:** turis dengan daftar "tempat-tempat yang belum saya kunjungi". Lingkungan baru? Prioritas utama. Kafe yang sudah Anda datangi lima puluh kali? Tidak menarik lagi.

Ini adalah ide lama yang bagus (MBIE-EB, UCB). Kelemahannya: di dunia yang besar atau kontinu, Anda tidak pernah mengunjungi status yang *persis* sama dua kali, sehingga hitungan mentahnya selalu 1 — itulah sebabnya jenis berikutnya ada.

### 2. Kebaruan kesalahan-prediksi (Prediction-error novelty): "Saya tidak menyangka *itu* akan terjadi" {#2-prediction-error-novelty-i-didnt-see-that-coming}

Ini adalah ide di balik **ICM** (Intrinsic Curiosity Module, Pathak et al. 2017) yang terkenal dan sepupunya **RND** (Random Network Distillation, Burda et al. 2018). Alih-alih menghitung, agen menyimpan sebuah **model yang mencoba memprediksi apa yang terjadi selanjutnya** — "jika saya di sini dan saya melakukan ini, di mana saya akan berakhir?" — dan memberi imbalan pada diri sendiri berdasarkan **seberapa salah model tersebut**:

```
bonus keingintahuan  =  kejutan  =  -log P( status yang sebenarnya saya capai | di mana saya tadi, apa yang saya lakukan )
```

- Situasi yang belum pernah dilihat model → ia memprediksi dengan buruk → kejutan besar → bonus besar → "pergi menjelajah ke sana!"
- Situasi yang telah dilihat model seratus kali → ia memprediksi dengan sempurna → nol kejutan → nol bonus → "sudah pernah ke sana, sudah paham, lanjut."

**Analogi kehidupan nyata:** seorang anak belajar bagaimana dunia bekerja dengan bermain. Menjatuhkan gelas dari meja untuk *pertama kalinya* sangat menarik (gelasnya pecah!). Keseratus kalinya, Anda sudah tahu gelas itu akan pecah — tidak menarik lagi. Keingintahuan = celah antara apa yang Anda harapkan dan apa yang terjadi.

Dalam kode tabular kami, "model" hanyalah tabel hitungan transisi, dan "seberapa salah model tersebut" adalah kejutan `-log P`. ICM/RND yang asli menggunakan jaringan saraf sehingga ide yang sama bekerja pada piksel mentah — tetapi prinsipnya identik.

> **Mengapa ada dua versi?** Berbasis hitungan sangat sederhana dan merupakan baseline yang bagus. Kesalahan-prediksi dapat diskalakan ke dunia besar yang tidak pernah berulang dan memberikan sinyal yang lebih *tajam*: di lingkungan deterministik, begitu Anda melihat transisi sekali, kejutannya langsung turun ke ~0, sedangkan bonus hitungan memudar secara perlahan sebesar `1/sqrt(N)`. Dalam eksperimen kami, agen prediction-error memecahkan MiniMontezuma dalam beberapa lusin episode; agen hitungan juga sampai di sana, hanya saja lebih lambat dan kurang andal.

## Apa yang Dilakukan Kode Kami {#what-our-code-does}

`curiosity_bonus.py` melatih sebuah **tabular Q-learner** biasa pada `MiniMontezumaEnv` — sebuah gridworld dua ruangan kecil di mana Anda harus berjalan ke sebuah **kunci**, mengambilnya (sekarang **pintu** terbuka), berjalan menembusnya, dan mencapai **harta karun**. Imbalan (+1) muncul *hanya* di harta karun, setelah ~15 langkah sempurna. Program ini menjalankan tiga agen dan memplotnya:

| Agen | Apa yang dilakukannya di MiniMontezuma |
|-------|-------------------------------|
| **epsilon-greedy (tanpa keingintahuan)** | Berkeliaran di dekat awal, *tidak pernah* mencapai kunci, skor tetap 0 selamanya. |
| **count-based bonus** | Berhasil menemukan kunci; melewati seluruh rangkaian ke harta karun mungkin ~40% episode. Berhasil — hanya sedikit berisik. |
| **prediction-error bonus** | Pertama kali mencapai kunci *dan* harta karun dalam ~20–25 episode; saat `beta` meluruh, ia konvergen untuk memecahkannya pada setiap episode. |

Gambar menunjukkan:
- kurva pembelajaran: *P(mencapai harta karun)* selama pelatihan,
- kurva kedua untuk milestone perantara *P(mengambil kunci)*,
- dan **heat-map kunjungan status** pada grid — agen tanpa keingintahuan adalah gumpalan ketat di dekat awal; agen yang ingin tahu membanjiri *kedua* ruangan.

## Mekanisme dalam Satu Gambar {#the-mechanism-in-one-picture}

```
         tanpa keingintahuan                   dengan bonus keingintahuan
   reward:  0 0 0 0 0 0 0 0 ... 0  (+1?)        0 0 0 0 0 0 0 0 ... 0  (+1!)
            └──── tidak ada yang dipelajari ──┘   └ + 0.4 0.3 0.9 0.2 ... ┘  (dibuat sendiri,
                                                  padat, menunjuk "ke arah hal baru")
   hasil:  jalan acak, tak pernah ketemu +1     menyisir dunia secara sistematis,
                                                 menemukan +1, lalu bonus memudar
```

Bonus keingintahuan mengubah *"Saya belum melihat ini"* menjadi imbalan, sehingga agen **sengaja mendorong ke wilayah yang belum dijelajahi** alih-alih meronta secara acak. Dan karena bonus menyusut saat hal-hal menjadi akrab (dan `beta` meluruh), begitu agen menemukan imbalan yang sebenarnya, ia secara alami berhenti membuang waktu dan mulai mengeksploitasi.

## Beberapa Peringatan Jujur {#a-few-honest-caveats}

- **Masalah "TV berisik" (noisy-TV problem).** Agen kesalahan-prediksi dapat terhipnotis oleh sumber keacakan murni (TV yang menayangkan statis, dadu yang menggelinding) — ia tidak akan *pernah* bisa memprediksinya, sehingga kejutannya tidak pernah pudar. Trik asli ICM adalah memprediksi dalam *ruang fitur yang dipelajari* yang mengabaikan hal-hal yang tidak dapat dikendalikan agen; RND menghindarinya dengan cara berbeda. Gridworld deterministik kami tidak memiliki TV berisik, jadi kita tidak menemui ini.
- **Keingintahuan adalah sarana, bukan tujuan.** Itulah sebabnya `beta` meluruh. Agen yang tetap penasaran secara maksimal selamanya tidak akan pernah menetap untuk benar-benar *menang*.
- **Penskalaan deep exploration masih sulit.** Bonus dalam imbalan membantu, tetapi Q-learning tabular biasa lambat untuk merambatkan optimisme yang dihasilkan ke seluruh rangkaian panjang (lihat `compare_exploration.py`). Memecahkan Montezuma versi piksel membutuhkan tenaga tambahan — RND dengan jaringan saraf, bootstrapped DQN, Go-Explore.

## Kata Kunci untuk Diingat {#key-words-to-remember}

| Kata | Makna |
|------|---------|
| **Intrinsic reward** | Imbalan yang dihasilkan agen untuk dirinya sendiri, terpisah dari imbalan lingkungan |
| **Extrinsic reward** | Imbalan nyata dari lingkungan (poin, menang/kalah) |
| **Sparse reward** | Imbalan bernilai nol hampir di mana-mana; Anda hanya mendapatkannya setelah urutan benar yang panjang |
| **Novelty / surprise** | Seberapa baru atau tidak terduga suatu status (atau transisi) — hal yang diberi imbalan oleh keingintahuan |
| **Count-based bonus** | Kebaruan ≈ `1/sqrt(visit count)` — bonus eksplorasi klasik |
| **ICM** | Intrinsic Curiosity Module: kebaruan = kesalahan prediksi model forward (dalam ruang fitur yang dipelajari) |
| **`beta`** | Bobot pada bonus keingintahuan; biasanya dilebur (annealed) menuju 0 sehingga agen akhirnya mengeksploitasi |

## Ringkasan Satu Kalimat {#one-sentence-summary}

> **Bonus keingintahuan adalah imbalan yang diberikan sendiri untuk kebaruan — ia menghasilkan sinyal padat "pergi-jelajahi-ke-sana" yang menyeret agen melalui dunia imbalan-jarang yang jika tidak, tidak akan pernah terpecahkan, lalu perlahan memudar setelah semuanya menjadi akrab.**
