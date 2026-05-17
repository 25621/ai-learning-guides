# Kebijakan Terkondisi-Tujuan (Goal-Conditioned Policy)

## Ide Besar: Satu Kebijakan untuk Semua {#the-big-idea-one-policy-to-rule-them-all}

Bayangkan Anda adalah seorang sopir pengantar barang. Anda tidak butuh keahlian yang berbeda untuk setiap alamat tujuan. Anda tahu cara mengemudi, membaca peta, dan menavigasi lalu lintas — Anda cukup memasukkan *tujuan hari ini* dan berangkat.

Sebuah **kebijakan terkondisi-tujuan (goal-conditioned policy)** bekerja dengan cara yang sama. Alih-alih melatih satu agen yang hanya bisa pergi ke satu tujuan tetap, kita melatih satu agen tunggal yang menerima tujuan apa pun sebagai input dan mencari tahu cara untuk sampai ke sana.

---

## Perbedaannya dengan RL Standar {#how-it-differs-from-standard-rl}

Dalam RL standar (seperti yang dibahas dalam fase-fase awal kurikulum), fungsi imbalannya sudah tertanam: "mencapai sel (7, 7), dapatkan +1." Agen belajar tepat satu hal: cara mencapai sel *tersebut*.

Dalam RL terkondisi-tujuan, imbalannya bergantung pada apakah agen mencapai *tujuan apa pun yang diberikan kali ini*. Kebijakan tersebut mempelajari:

> **"Mengingat di mana saya sekarang dan ke mana saya ingin pergi, apa yang harus saya lakukan?"**

Tujuan (goal) ikut *bersama* agen, seperti tujuan yang diketikkan ke dalam aplikasi navigasi.

---

## Masalah Imbalan Jarang (Sparse Reward) {#the-sparse-reward-problem}

Inilah masalahnya: belajar dari imbalan jarang (hanya +1 di tujuan, 0 di tempat lain) sangatlah sulit. Sebagian besar upaya gagal — agen berkeliaran secara acak, tidak pernah membentur tujuan, dan jaringan tidak mendapatkan informasi berguna untuk dipelajari.

Bayangkan mencoba belajar melempar dart dengan mata tertutup. Anda melempar seribu kali dan selalu meleset. Setelah seribu kegagalan, Anda masih tidak tahu seperti apa rasanya "lemparan yang bagus".

Di sinilah **Hindsight Experience Replay (HER)** berperan.

---

## Hindsight Experience Replay: Belajar dari Kegagalan {#hindsight-experience-replay-failing-forward}

Trik HER sangat sederhana dan indah. Setelah episode yang gagal, HER bertanya:

> *"Meskipun Anda tidak mencapai tujuan... di mana Anda sebenarnya berakhir?"*

Ia kemudian **memutar ulang episode yang sama**, tetapi berpura-pura bahwa posisi akhir agen yang sebenarnya **adalah** tujuannya sedari awal. Tiba-tiba, episode yang gagal menjadi episode yang sukses — untuk tujuan yang berbeda.

Ini seperti pemain basket yang gagal dan terus menembak ke ring namun selalu meleset. HER akan berkata: "Oke, kamu selalu mengenai tembok kiri. Selamat — kamu hebat dalam mengenai tembok kiri! Ayo catat lemparan itu sebagai upaya sukses mengenai tembok kiri." Seiring waktu, pemain tersebut membangun keterampilan dalam mengenai target *apa pun*, dan akhirnya mentransfer keterampilan itu ke ring basket yang sebenarnya.

Ini mengubah ribuan "kegagalan" menjadi pustaka kaya akan navigasi *sukses* ke berbagai tempat yang berbeda. Agen belajar untuk mencapai semuanya, yang menggeneralisasi ke target yang sebenarnya.

---

## Analogi Kehidupan Nyata: Balita Belajar Menyusun Balok {#the-real-life-analogy-toddler-learning-to-stack-blocks}

Seorang balita yang mencoba memasukkan balok ke dalam ember sering kali meleset. Tetapi setiap "meleset" itu mendaratkan balok di *suatu tempat*. Jika Anda memutar ulang setiap kegagalan sebagai "kamu tadi mencoba meletakkannya *tepat di sana* — dan kamu berhasil!", balita tersebut membangun keterampilan motorik halus di seluruh meja. Segera mereka bisa meletakkan balok di mana saja — termasuk di dalam ember.

---

## Apa yang Dilakukan Kode Kami {#what-our-code-does}

Skrip `goal_conditioned_policy.py` berjalan dalam **labirin 7x7** dengan dinding. Di awal setiap episode, sel tujuan acak dipilih. Agen harus menemukannya.

Kebijakan tersebut mengambil dua input di setiap langkah:
1. Di mana agen berada saat ini
2. Ke mana ia ingin pergi

Setelah setiap episode (sukses atau tidak), HER menghasilkan beberapa "kesuksesan" sintetis tambahan dengan memberi label ulang posisi yang sebenarnya dikunjungi sebagai tujuan alternatif.

Pelatihan berjalan selama 3.000 episode dengan tingkat eksplorasi yang meluruh — agen bereksplorasi lebih banyak di awal dan kemudian semakin percaya pada apa yang telah ia pelajari.

---

## Apa yang Ditunjukkan Grafik {#what-the-charts-show}

![Hasil Kebijakan Terkondisi-Tujuan](outputs/goal_conditioned_policy.png)

**Kiri — Tingkat Keberhasilan Selama Pelatihan:** Setiap episode bisa berupa keberhasilan (mencapai tujuan) atau kegagalan. Kurva naik dengan stabil seiring dengan meningkatnya keterampilan navigasi universal agen. Di akhir, agen mencapai tujuan apa pun hampir setiap saat.

**Kanan — Heatmap Tingkat Keberhasilan Tujuan:** Setelah pelatihan, kami menguji agen pada setiap kemungkinan sel tujuan dan mewarnai setiap sel berdasarkan seberapa sering agen mencapainya. Hijau berarti agen dapat diandalkan mencapai titik tersebut; merah berarti ia masih kesulitan. Agen yang terlatih dengan baik menunjukkan warna hijau di hampir seluruh labirin.

---

## Di Mana Ini Muncul di Dunia Nyata {#where-this-shows-up-in-the-real-world}

| Aplikasi | "Tujuan"-nya |
|-------------|------------|
| Jangkauan lengan robot | Posisi target 3D |
| Mobil otonom | Koordinat GPS |
| Model bahasa asisten | Instruksi pengguna |
| Karakter Game (NPC) | Titik arah mana pun di peta |

Kebijakan terkondisi-tujuan adalah salah satu blok bangunan utama untuk HIRO (Hierarchical RL with subgoals) — manajer tingkat tinggi memilih sub-tujuan, dan pekerja tingkat rendah adalah jenis kebijakan terkondisi-tujuan seperti ini.

---

## Ringkasan Satu Kalimat {#one-sentence-summary}

> **Kebijakan terkondisi-tujuan adalah satu agen yang dapat bernavigasi ke tujuan mana pun — dan HER memungkinkan belajar dari kegagalan dengan berpura-pura setiap tembakan yang meleset ditujukan ke mana pun ia mendarat.**
