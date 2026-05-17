# Matrix Games: Dunia Multi-Agen Paling Sederhana 🎲

## Apa Itu Matrix Game?

Bayangkan Anda dan seorang teman masing-masing memilih tanda tangan — **batu, kertas, atau gunting** — *pada saat yang sama*. Anda tidak melihat pilihan satu sama lain. Pemenangnya ditentukan oleh tabel kecil:

|        | Batu | Kertas | Gunting |
|--------|:----:|:-----:|:--------:|
| Batu     |  0,0  | -1,+1 | +1,-1 |
| Kertas    | +1,-1 |  0,0  | -1,+1 |
| Gunting | -1,+1 | +1,-1 |  0,0  |

Tabel itu adalah *seluruh dunia* dari permainan tersebut. Tidak ada gerakan, tidak ada waktu, tidak ada peta. Hanya keputusan satu kali. Kita menyebut ini **matrix game** karena matriks imbalan (payoff matrix) adalah seluruh lingkungannya.

Matrix games adalah tempat paling bersih untuk mempelajari **RL multi-agen**, karena satu-satunya hal yang dapat berubah selama pelatihan adalah *kebijakan (policy)* masing-masing pemain — probabilitas memilih setiap tindakan.

---

## Mengapa Ini "Multi-Agen"

Dalam RL agen-tunggal, lingkungannya tetap: angin selalu bertiup ke arah yang sama, lantai tidak pernah bergerak. Agen berkembang dan akhirnya menang.

Dalam matrix game, "lingkungan" Anda adalah *agen pembelajar lainnya*. Seiring mereka bertambah pintar, apa yang dianggap sebagai langkah baik bagi Anda *berubah*. Ini disebut **non-stasioneritas (non-stationarity)**, dan ini adalah masalah utama dari RL multi-agen.

> Jika Anda terus memainkan Batu, lawan Anda akhirnya akan mulai selalu memainkan Kertas. Jadi Anda beralih ke Gunting. Jadi mereka beralih ke Batu. Jadi Anda beralih ke Kertas... dan seterusnya. "Langkah terbaik" tidak pernah tetap diam.

Solusi klasiknya adalah **strategi campuran (mixed strategies)**: jangan pilih satu tindakan pun secara deterministik — acaklah sedemikian rupa sehingga lawan tidak dapat mengeksploitasi.

---

## Tiga Permainan yang Kita Mainkan

### 1) Batu-Kertas-Gunting (zero-sum)
- Keuntungan satu pemain adalah kerugian pemain lain.
- **Kesetimbangan Nash (Nash equilibrium)** adalah: setiap pemain memilih setiap tindakan dengan probabilitas ⅓. Penyimpangan apa pun dapat dieksploitasi.
- Kita mengharapkan kedua pembelajar-Q kita bergoyang di sekitar ⅓-⅓-⅓ — tidak pernah benar-benar stabil, karena setiap kali yang satu bergeser, yang lain bereaksi.

### 2) Dilema Tahanan (Prisoner's Dilemma) (general-sum)
Dua tersangka diinterogasi secara terpisah:

|           | Bekerja Sama | Berkhianat |
|-----------|:---------:|:------:|
| Bekerja Sama |   3, 3    |  0, 5  |
| Berkhianat    |   5, 0    |  1, 1  |

- "Berkhianat" mengalahkan "Bekerja Sama" apa pun yang dilakukan pihak lain — ini adalah **strategi dominan**.
- Kedua pemain rasional → keduanya berkhianat → keduanya mendapat 1, meskipun (Bekerja Sama, Bekerja Sama) masing-masing mendapat 3. Respons terbaik yang egois menghancurkan kesejahteraan kelompok.
- Kita mengharapkan Q-learning konvergen secara bersih ke (Berkhianat, Berkhianat).

### 3) Stag Hunt (koordinasi)
Dua pemburu bersama-sama dapat menjatuhkan rusa jantan (hadiah besar), atau masing-masing puas dengan seekor kelinci (hadiah kecil tapi aman):

|       | Rusa | Kelinci |
|-------|:----:|:----:|
| Rusa  | 4, 4 | 0, 3 |
| Kelinci  | 3, 0 | 2, 2 |

- (Rusa, Rusa) adalah **payoff-dominant** — terbaik bagi keduanya.
- (Kelinci, Kelinci) adalah **risk-dominant** — aman jika Anda tidak mempercayai pasangan Anda.
- Hasil tergantung pada kondisi awal: pembelajar-Q independen seringkali berakhir pada kesetimbangan yang *lebih buruk* (Kelinci, Kelinci) karena kelinci lebih aman untuk dipelajari.

---

## Contoh Kehidupan Nyata

- **Penetapan harga dalam duopoli.** Dua kedai kopi di jalan yang sama masing-masing memilih harga setiap pagi. Bentuk matriks imbalan menentukan apakah mereka berakhir pada harga "kooperatif" yang tinggi (baik bagi mereka, buruk bagi pelanggan) atau harga rendah yang saling menjatuhkan.
- **Protokol jaringan.** Router dan pengirim memilih strategi waktu; hasil kemacetan jaringan ditentukan oleh imbalan seperti matrix-game antara berhasil lewat vs. mundur.
- **Penawaran dalam lelang.** Setiap penawar memilih tawaran tanpa mengetahui penawar lainnya; imbalan tergantung pada seluruh vektor tawaran. Kesetimbangan Nash adalah *strategi penawaran*, bukan satu angka saja.

---

## Apa yang Dilakukan Kode Kami

Untuk setiap permainan kita:
1. Membuat dua pembelajar-Q tanpa status (Q hanyalah satu angka per tindakan — tidak ada status dalam permainan satu kali/1-shot).
2. Melakukan loop selama 20.000 langkah. Setiap langkah: kedua agen memilih tindakan ε-greedy secara simultan, mendapatkan imbalan, memperbarui nilai-Q mereka.
3. Melacak **frekuensi tindakan empiris** masing-masing agen dalam jendela 500 langkah bergulir. Alih-alih hanya melihat probabilitas abstrak, kita menghitung tindakan apa yang sebenarnya mereka pilih baru-baru ini (misalnya, "dalam 500 putaran terakhir, mereka memainkan Batu 40% dari waktu"). Ini memberi kita gambaran praktis real-time dari perubahan strategi mereka.
4. Membuat plot frekuensi seiring waktu, simpan ke `outputs/<game>.png`, dan cetak nilai-Q akhir.

### Apa yang seharusnya Anda lihat

| Game | Hasil yang diharapkan dari plot |
|------|------------------------------|
| **Batu-Kertas-Gunting** | Kedua pemain berada di dekat ⅓-⅓-⅓ tetapi bergetar secara nyata. Kurva saling mengejar — perilaku siklus klasik. |
| **Dilema Tahanan** | Frekuensi "Berkhianat" kedua pemain naik ke ~1,0 dengan cepat. "Bekerja Sama" hancur. |
| **Stag Hunt** | Sebagian besar seed acak menetap pada (Kelinci, Kelinci). Beberapa seed beruntung mencapai (Rusa, Rusa) — coba ubah seed dalam skrip dan lihat perubahannya. |

---

## Di Mana Pembelajaran Independen Gagal

Agen kita bersifat *independen* — mereka hanya melihat imbalan mereka sendiri, tidak pernah melihat tindakan atau nilai-Q lawan. Ini adalah baseline paling sederhana dan memiliki batas:

- Ia **tidak dapat menjamin konvergensi** dalam permainan general-sum.
- Ia dapat terjebak dalam **kesetimbangan yang buruk** (Stag Hunt).
- Ia **tidak dapat memodelkan lawan**.

Algoritma multi-agen yang sebenarnya memperbaiki hal ini dengan secara eksplisit menalar tentang pembelajar lain. Berikut adalah apa yang dilakukan masing-masing, dalam bahasa sederhana:

| Algoritma | Ide utama | Analogi kehidupan nyata |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| **Fictitious play** | Menyimpan catatan terus-menerus tentang seberapa sering lawan Anda memilih setiap tindakan. Asumsikan besok mereka akan melakukan apa yang selalu mereka lakukan — lalu pilih respons terbaik Anda sendiri terhadap keyakinan itu. | Memperhatikan kebiasaan lawan selama banyak permainan catur dan menyesuaikan pembukaan Anda. |
| **CFR (Counterfactual Regret Minimisation)** | Setelah setiap putaran, tanyakan *"Seberapa besar penyesalan saya karena tidak memilih setiap tindakan lainnya?"* Secara bertahap geser probabilitas ke arah tindakan yang Anda sesali karena terlewatkan. Digunakan dalam poker karena menangani permainan **informasi-tidak-sempurna** (Anda tidak melihat kartu lawan). | Setelah satu putaran poker, memutar ulang dan berpikir: *"Saya seharusnya bertaruh lebih banyak — saya akan melakukannya lain kali."* |
| **LOLA (Learning with Opponent-Learning Awareness)** | Langkah gradien Anda memperhitungkan fakta bahwa lawan *juga* melakukan langkah gradien. Anda mengoptimalkan pembaruan Anda sendiri sambil mengantisipasi pembaruan lawan berikutnya — dua langkah ke depan alih-alih satu. | Menegosiasikan kesepakatan sambil berpikir: *"Jika saya menawarkan X, mereka akan membalas dengan Y, jadi saya harus mulai dengan Z."* |
| **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)** | *Kritikus* (estimasi nilai) setiap agen dilatih dengan **pandangan global**: ia melihat observasi dan tindakan semua orang. *Aktor* (kebijakan yang dijalankan) tetap hanya menggunakan informasi lokal — ini adalah pola CTDE (Centralized Training with Decentralized Execution). | Pelatih bola basket yang menonton seluruh lapangan (kritikus terpusat) tetapi mengajar setiap pemain untuk bereaksi hanya pada apa yang dapat mereka lihat (aktor terdesentralisasi). |

Tetapi Q-learning independen adalah langkah pertama yang tepat. Anda melihat masalah non-stasioneritas langsung di depan mata, dan perbaikannya masuk akal setelahnya.

---

## Kata Kunci untuk Diingat

| Kata | Makna |
|------|---------|
| **Payoff matrix** | Tabel yang mendefinisikan permainan multi-agen satu kali (1-shot) |
| **Kesetimbangan Nash** | Profil kebijakan di mana tidak ada agen tunggal yang dapat meningkat dengan menyimpang |
| **Strategi campuran** | Kebijakan yang mengacak beberapa tindakan |
| **Non-stasioneritas** | Lingkungan (= agen lain) terus berubah saat ia belajar |
| **Pembelajar independen** | Agen yang mengabaikan keberadaan pembelajar lain |
| **Zero-sum** | Keuntungan satu agen adalah kerugian agen lainnya secara persis |
| **General-sum** | Kedua agen bisa menang, keduanya bisa kalah, atau apa pun di antaranya |

---

## Ringkasan Satu Kalimat {#one-sentence-summary}

> **Dalam matrix games, "lingkungan" adalah pembelajar lain — sehingga langkah terbaik terus bergerak.**

Ini adalah ide dasar di balik setiap algoritma multi-agen yang akan Anda temui nanti, dari self-play hingga MADDPG hingga MARL dengan komunikasi.
