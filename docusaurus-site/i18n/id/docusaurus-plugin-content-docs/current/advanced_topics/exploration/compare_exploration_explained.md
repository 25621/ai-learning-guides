# Membandingkan Strategi Eksplorasi 🔦

## Masalah dalam Satu Kalimat {#the-one-sentence-problem}

Agen RL harus melakukan dua hal yang menarik ke arah yang berlawanan:

- **Eksploitasi (Exploit)**: lakukan hal yang paling berhasil sejauh ini.
- **Eksplorasi (Explore)**: coba sesuatu yang baru, kalau-kalau itu bahkan lebih baik.

Terlalu condong ke eksploitasi dan Anda akan dengan senang hati menetap pada rutinitas yang biasa-biasa saja selamanya. Terlalu condong ke eksplorasi dan Anda tidak pernah mendapatkan keuntungan. *Bagaimana* Anda bereksplorasi — bukan hanya *apakah* — adalah apa yang membedakan agen yang memecahkan Montezuma's Revenge dari yang mencetak skor nol.

Skrip ini mengadu **lima** strategi eksplorasi satu sama lain pada dua tugas sulit yang sama, sehingga Anda dapat melihat karakteristik mereka masing-masing.

## Analogi Kehidupan Nyata: Memilih Tempat Makan Siang {#real-life-analogy-picking-a-lunch-spot}

Anda baru saja pindah ke kota baru dengan 200 restoran.

- **ε-greedy** = "Pergi ke favorit saya saat ini, tetapi sekali setiap sepuluh hari lempar dadu dan pilih restoran *secara acak*." Anda akan mencicipi secara luas tetapi *tanpa tujuan* — dan Anda akan terus mendatangi tempat-tempat yang sudah Anda benci.
- **Optimistic initialisation** = "Asumsikan *setiap* restoran yang belum saya coba adalah yang terbaik di kota sampai terbukti sebaliknya." Anda akan secara metodis mencoba semua 200 restoran, mencoret masing-masing saat kenyataan mengecewakan Anda — dan Anda akan menemukan yang benar-benar hebat dengan cepat.
- **UCB (Upper Confidence Bound)** = "Pilih favorit saya, tetapi berikan *bonus* untuk tempat-tempat yang jarang saya coba — semakin sedikit saya tahu tentangnya, semakin besar bonusnya." Ini cerdas tentang tempat yang tidak dikenal *mana* yang harus dicoba hari ini, tetapi setiap keputusan bersifat lokal: ia memilih opsi yang tampak terbaik *saat ini* tanpa merencanakan rute melalui seluruh lingkungan yang belum dijelajahi. Ia tidak akan berpikir "Saya harus menyeberangi kota ke sisi timur, karena ada dua puluh tempat yang belum dicoba berkumpul di sana" — setiap restoran dievaluasi secara terpisah, langkah demi langkah.
- **Count-based reward bonus** = seperti UCB, tetapi Anda juga *menikmati kebaruan itu sendiri* — makan di tempat yang baru memberikan kepuasan tersendiri, dan kepuasan itu membentuk rencana jangka panjang Anda tentang lingkungan mana yang akan dijelajahi.
- **Prediction-error reward bonus** = "Saya merasa senang dari makanan yang *mengejutkan* saya — sesuatu yang tidak bisa saya prediksi." Tempat baru yang ternyata persis seperti yang diharapkan? Biasa saja. Tempat yang sangat berbeda dari model mental Anda? Menarik, dan Anda memperbarui rencana Anda untuk mencari lebih banyak tempat seperti itu.

## Lima Strategi (semua ada di `compare_exploration.py`) {#the-five-strategies-all-in-compare_explorationpy}

### 1. ε-greedy — default, dan ini adalah *kegamangan*, bukan eksplorasi {#1-ε-greedy--the-default-and-its-dithering-not-exploring}

Bertindak secara rakus (greedy), tetapi dengan probabilitas ε ambil tindakan acak secara seragam. Ini adalah baseline standar dalam DQN dan kawan-kawan. Kelemahan fatalnya pada tugas-tugas sulit: **setiap langkah adalah lemparan koin yang independen.** Untuk berhasil melalui rangkaian `N` langkah yang benar, Anda membutuhkan koin untuk mendarat tepat `N` kali berturut-turut — itu sangat tidak mungkin secara eksponensial. ε-greedy adalah *getaran*, bukan *eksplorasi*.

### 2. Optimistic initialisation — "tidak bersalah sampai terbukti membosankan" {#2-optimistic-initialisation--innocent-until-proven-boring}

Mulai *setiap* nilai-Q pada return terbesar yang mungkin, `R_max / (1 − γ)`. Sekarang tindakan yang belum pernah dicoba agen terlihat seperti hal terbaik di dunia, sehingga kebijakan **greedy** dipaksa untuk mencobanya; hanya setelah mengunjunginya nilai tersebut turun menuju kebenaran. Optimisme tentang wilayah yang *belum* dicoba secara otomatis **merambat melalui fungsi nilai** (melalui bootstrap Q-learning), sehingga agen ditarik, langkah demi langkah, menuju bagian dunia yang belum pernah ia lihat. Hampir gratis, tanpa pembukuan ekstra — dan, seperti yang akan Anda lihat, penjelajah *deep* terkuat dalam dunia tabular kecil.

### 3. Pemilihan tindakan gaya UCB — bonus dalam *pilihan*, bukan dalam *imbalan* {#3-ucb-style-action-selection--bonus-in-the-choice-not-the-reward}

Pilih `argmax_a [ Q(s,a) + c·√(ln t / N(s,a)) ]`: pilih tindakan bernilai tinggi, tetapi tiupkan nilai pada tindakan yang jarang Anda coba. Terkenal dari multi-armed bandits. Masalahnya: bonus hanya ada dalam **aturan pemilihan tindakan**, tidak pernah dalam imbalan — sehingga ia *tidak* mengalir melalui fungsi nilai. UCB hebat dalam "pastikan saya sudah mencoba setiap tindakan di status *ini*" tetapi lemah dalam "merencanakan rute menuju wilayah jauh yang belum dijelajahi."

### 4. Count-based **reward** bonus — keingintahuan, versi klasik {#4-count-based-reward-bonus--curiosity-the-classic-version}

Tambahkan `1/√(N(s,a))` ke **imbalan** (dengan bobot `beta` yang meluruh). Karena ada dalam imbalan, Q-learning *akan* merambatnya: status yang mengarah ke wilayah baru menjadi berharga. Ini adalah ide MBIE-EB / "exploration bonus" klasik.

### 5. Prediction-error **reward** bonus — keingintahuan, versi ICM/RND {#5-prediction-error-reward-bonus--curiosity-the-icmrnd-version}

Tambahkan `−log P(s'|s,a)` dari model forward kecil yang dipelajari ke imbalan (sekali lagi dengan `beta` yang meluruh). Sinyal kebaruan paling tajam dari kelimanya: dalam dunia deterministik, kejutan dari transisi turun ke ~0 saat Anda melihatnya sekali, alih-alih memudar perlahan seperti `1/√N`. Sepupu tabular dari ICM / RND.

## Dua Tugas Pengujian {#the-two-test-tasks}

- **Tugas A — MiniMontezuma**: gridworld kunci→pintu→harta karun, imbalan hanya ada di harta karun (~15 langkah sempurna jauhnya). Menguji "dapatkah Anda bertahan dalam rangkaian imbalan jarang (sparse-reward) yang panjang sama sekali?"
- **Tugas B — DeepSea(N)**: rangkaian deep-exploration buku teks, dijalankan pada panjang `N = 5, 8, 11, 14`. Imbalan bersembunyi di balik `N` langkah yang benar, masing-masing dengan biaya langsung kecil — sehingga agen yang picik belajar untuk menghindari biaya tersebut dan tidak pernah menemukan hadiahnya. Menguji "apakah strategi Anda masih berfungsi saat rangkaian semakin *panjang*?"

## Apa yang Sebenarnya Terjadi (jalankan dan lihat) {#what-actually-happens-run-it-and-see}

**Tugas A — MiniMontezuma:**

| Strategi | Harta karun pertama | Tingkat penyelesaian akhir |
|----------|---------------:|-----------------:|
| ε-greedy | tidak pernah | 0.00 |
| optimistic init | ~episode 1 | 1.00 |
| UCB action selection | ~episode 3 | ~0.95 |
| count reward bonus | ~episode 82 | ~0.41 |
| prediction reward bonus | ~episode 23 | 1.00 |

**Tugas B — DeepSea, fraksi seed yang menemukan imbalan:**

| Strategi | N=5 | N=8 | N=11 | N=14 |
|----------|----:|----:|-----:|-----:|
| ε-greedy | 0 | 0 | 0 | 0 |
| optimistic init | 1.0 | 1.0 | 1.0 | 1.0 |
| UCB action selection | 1.0 | 1.0 | 0.0 | 0.0 |
| count reward bonus | 1.0 | 1.0 | ~0.1 | 0.0 |
| prediction reward bonus | ~0.9 | ~0.8 | ~0.9 | ~0.2 |

*(Angka-angka ini sedikit berfluktuasi dengan seed acak, tetapi polanya sangat jelas.)*

## Pelajaran {#the-lessons}

1. **ε-greedy bukan eksplorasi.** Ia tidak pernah menyelesaikan tugas sulit *apa pun*. Kegamangan acak tidak akan pernah bisa menembus rangkaian panjang yang benar. (Namun ini masih menjadi default di banyak kode — karena pada tugas-tugas *mudah* ia sudah cukup baik dan sangat sederhana.)

2. **Eksplorasi nyata berarti bersikap optimis tentang hal yang tidak diketahui — dengan satu atau lain cara.** Baik Anda menanamkan optimisme ke dalam *nilai awal* (strategi 2), ke dalam *pilihan tindakan* (strategi 3), atau ke dalam *imbalan yang dihasilkan sendiri* (strategi 4–5), benang merahnya adalah: *buat yang belum dijelajahi terlihat menarik*, lalu biarkan pembelajaran membawa Anda ke sana.

3. **Pada kisi (grid) dengan imbalan jarang, keempat strategi "nyata" berhasil — dan bonus prediction-error sampai di sana paling cepat**, karena ia menghasilkan sinyal "ini baru" yang paling tajam.

4. **Pada rangkaian yang *dalam*, di mana optimisme harus menempuh perjalanan jauh, pemenang sederhananya adalah optimistic initialisation.** Ia merambat optimisme melalui fungsi nilai secara gratis. UCB berantakan lebih dulu (bonusnya tidak pernah masuk ke fungsi nilai, sehingga ia tidak bisa *merencanakan* secara mendalam). Bonus imbalan lebih baik — mereka *merambat* — tetapi Q-learning tabular biasa lambat untuk mendorong optimisme itu ke seluruh rangkaian panjang sebelum bonus meluruh.

5. **Poin terakhir itulah alasan mengapa penskalaan deep exploration ke piksel membutuhkan tenaga tambahan** — bootstrapped DQN, RND dengan jaringan saraf asli (sehingga optimisme *menggeneralisasi* di seluruh status serupa alih-alih merambat satu sel pada satu waktu), Go-Explore (secara harfiah mengingat dan kembali ke status yang menjanjikan). Mainan tabular di sini menunjukkan kepada Anda *prinsip-prinsipnya*; sistem yang sebenarnya adalah prinsip-prinsip yang sama ditambah jaringan yang menggeneralisasi.

## Kata Kunci untuk Diingat {#key-words-to-remember}

| Kata | Makna |
|------|---------|
| **Exploration–exploitation trade-off** | Mencoba hal baru vs mendapatkan hasil dari apa yang Anda ketahui — ketegangan utama dalam RL |
| **Dithering** | "Eksplorasi" dengan menambahkan noise acak ke tindakan (ε-greedy, noise Gaussian) — lemah pada tugas sulit |
| **Optimism in the face of uncertainty** | Prinsip payung: perlakukan yang tidak diketahui seolah-olah hebat sampai Anda memeriksanya |
| **Optimistic initialisation** | Mengimplementasikan prinsip tersebut dengan memulai semua nilai pada return maksimum yang mungkin |
| **UCB** | Upper Confidence Bound: pilih `argmax (nilai + bonus yang menyusut seiring jumlah kunjungan)` |
| **Deep exploration** | Eksplorasi yang membutuhkan urutan tindakan "tidak biasa" yang *koheren* dan panjang, bukan hanya satu |
| **Peleburan (annealing) `beta`** | Menyusutkan bobot keingintahuan seiring waktu sehingga agen akhirnya berhenti bereksplorasi dan mengeksploitasi |

## Ringkasan Satu Kalimat {#one-sentence-summary}

> **ε-greedy hanyalah noise; setiap strategi eksplorasi nyata bekerja dengan membuat yang belum dijelajahi terlihat menarik — melalui nilai-nilai optimis, bonus pilihan tindakan, atau imbalan kebaruan yang dihasilkan sendiri — dan pilihan yang tepat tergantung pada apakah imbalan Anda hanya *jarang* (seperti menemukan satu hadiah tersembunyi di lapangan datar) atau benar-benar *dalam* (seperti kunci kombinasi yang memerlukan urutan pilihan spesifik yang panjang dan tepat untuk dipecahkan).**
