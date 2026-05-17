# Self-Play: Mengajari Agen dengan Membiarkannya Melawan Diri Sendiri ♟️

## Apa Itu Self-Play?

Bayangkan seorang anak yang ingin menjadi sangat hebat dalam catur tetapi tidak punya teman bermain. Jadi ia bermain melawan dirinya sendiri. Tangan kiri melawan tangan kanan. Setiap permainan, *kedua* sisi mencoba menang. Setiap permainan, *kedua* sisi belajar apa yang berhasil.

Itulah **self-play**: agen tunggal bertindak sebagai kedua pemain, dan setiap langkah menjadi pelajaran bagi siapa pun yang bergerak selanjutnya. Tanpa guru, tanpa lawan ahli. Hanya seorang pembelajar yang juga menjadi tangganya sendiri.

Self-play terdengar seperti tipuan — bukankah Anda butuh lawan sungguhan? — tetapi ini adalah mesin di balik pencapaian RL paling terkenal dalam dekade terakhir: **AlphaGo Zero**, **AlphaZero**, **MuZero**, **OpenAI Five**. Semuanya menggunakan self-play. Alasannya sederhana: seiring Anda meningkat, lawan Anda juga meningkat dalam jumlah yang sama. Tantangannya selalu sesuai dengan tingkat keahlian Anda.

---

## Mengapa Ini Berhasil

Tiga hal membuat self-play istimewa:

1. **Lawan tanpa batas.** Anda tidak pernah kehabisan permainan. Lawan selalu ada dan gratis.
2. **Kurikulum yang tumbuh bersama Anda.** Seorang pemula hanya bisa bermain melawan pemula lainnya. Seiring Anda menjadi lebih baik, bayangan Anda juga meningkat — secara otomatis.
3. **Simetri.** Dalam permainan zero-sum (kemenangan satu pemain adalah kekalahan pemain lain), satu set nilai-Q mendeskripsikan kedua sisi; Anda cukup membalik tandanya saat giliran pemain lain. Jadi *satu* tabel-Q dapat mengajar dirinya sendiri.

Tic-tac-toe adalah testbed yang sempurna: cukup kecil untuk muat dalam sebuah dictionary, tetapi cukup kompleks sehingga memilih langkah secara acak hampir selalu akan menyebabkan kekalahan melawan pemain yang strategis.

---

## Analogi Kehidupan Nyata

- **Melatih tenis melawan tembok.** Anda tidak bisa kalah melawan tembok, tetapi Anda bisa melatih servis Anda. Self-play adalah melakukan hal ini di kedua ujungnya — Anda adalah tembok *sekaligus* pemainnya, dan Anda berganti peran bolak-balik.
- **Klub debat yang memperdebatkan kedua sisi.** Pendebat yang lebih baik muncul dari kebiasaan selalu membela pandangan yang berlawanan dengan apa yang mereka yakini secara pribadi. Setiap argumen melatih serangan sekaligus pertahanan.
- **AlphaGo Zero.** Ia belajar dari nol permainan manusia. Dimulai dari gerakan acak, ia memainkan jutaan permainan melawan dirinya sendiri; dalam beberapa hari ia lebih baik daripada program Go mana pun sebelumnya, termasuk yang mengalahkan Lee Sedol.

---

## Apa yang Dilakukan Kode Kami

Kita mempelajari satu tabel-Q untuk *pemain yang saat ini akan bergerak*:

```
Q[(papan, pemain_yang_bergerak)][tindakan] = return yang diharapkan untuk pemain tersebut
```

Loop pelatihannya adalah:

1. Mulai papan kosong. `player = X`.
2. Kedua pemain bertindak dengan **agen yang sama**, menggunakan ε-greedy.
3. Setelah setiap permainan, telusuri mundur setiap triple (papan, pemain, tindakan) dalam sejarah dan terapkan pembaruan Q-learning.
4. Imbalan membalik tanda di setiap giliran: jika X menang, setiap langkah yang dilakukan X mendapat +1 (atau nilai bootstrap dari status kemenangan di masa depan); setiap langkah yang dilakukan O mendapat -1.
5. Kita perlahan mengurangi tingkat eksplorasi (ε) dari 0,2 → 0,02, sehingga agen berkomitmen pada permainan terbaiknya di akhir pelatihan alih-alih mencoba gerakan acak.

Setiap 2.500 episode kita mengevaluasi agen melawan **lawan acak** (kita membekukan proses pembelajaran sehingga tidak ada pembaruan baru yang dilakukan pada tabel-Q selama evaluasi, dan kedua belah pihak bermain secara rakus). Agen harus menang atau seri ~100% dari permainan tersebut setelah self-play yang cukup.

### Apa yang seharusnya Anda lihat

Setelah 50.000 episode self-play:

| Pertandingan | Hasil yang diharapkan |
|----------|-----------------|
| Agen terlatih vs Lawan acak (1000 permainan) | **~95-99% menang atau seri**, hampir 0% kalah |
| Agen terlatih vs Dirinya sendiri (200 permainan rakus) | **Semua 200 seri**. Tic-tac-toe adalah permainan yang selalu berakhir seri jika kedua pemain bermain dengan sempurna. Fakta bahwa self-play menghasilkan seri di setiap permainan adalah tanda konvergensi. |

Plot `outputs/self_play_tic_tac_toe.png` menunjukkan fraksi menang/seri/kalah agen melawan lawan acak seiring waktu:
- Tingkat kemenangan dimulai ~60% (ketika kedua pemain bermain secara acak, pemain pertama memiliki keuntungan inheren karena mereka dapat menempatkan lebih banyak penanda di papan, yang menyebabkan tingkat kemenangan baseline sekitar 60% untuk pemain X).
- Naik hingga >90%.
- Tingkat kekalahan turun hingga hampir 0%.

Skrip tersebut juga mencetak contoh permainan langkah-demi-langkah di akhir sehingga Anda dapat melihat agen bermain.

---

## Perhatikan Hal-Hal Halus Berikut

- **Pembalikan tanda itu penting.** Bug yang umum terjadi: lupa bahwa "lawan yang memaksimalkan nilai mereka" berarti *meminimalkan nilai kita* dalam target bootstrap. Pembaruan dalam kode kita menggunakan `target = imbalan - gamma * max(Q[berikutnya, lawan])`.
- **Simetri tidak dieksploitasi di sini.** Implementasi yang sebenarnya akan meng-kanonik-kan papan (artinya mereka akan memutar atau mencerminkan status papan apa pun ke dalam 'bentuk normal' standar yang unik sehingga agen mengenali situasi papan yang identik) untuk berbagi nilai-Q di 8 simetri. Kita melewati ini — ruang statusnya cukup kecil untuk di-brute-force.
- **Tabel-Q bertumbuh.** Setelah 50 ribu permainan self-play, Anda akan melihat beberapa ribu kunci status-pemain. Itu tidak masalah di sini; untuk catur atau Go Anda akan membutuhkan jaringan saraf sebagai gantinya, itulah sebabnya **AlphaZero mengganti tabel dengan CNN + MCTS**.

---

## Di Mana Self-Play Gagal

- **Permainan non-zero-sum.** "Kedua belah pihak senang" tidak kompatibel dengan permainan simetris; Anda tidak bisa begitu saja membalik tandanya.
- **Peran asimetris.** Jika "penyerang" dan "penahan" memiliki ruang tindakan yang berbeda, Anda membutuhkan dua jaringan terpisah.
- **Siklus strategi.** Self-play murni dapat terjebak dalam siklus seperti batu-kertas-gunting. AlphaStar memperbaiki hal ini dengan menyimpan kumpulan besar (*pool* atau "liga") versi agen masa lalu yang disimpan dan memilih lawan dari kumpulan tersebut secara acak, sehingga agen belajar untuk mengalahkan banyak gaya bermain yang berbeda daripada hanya yang sekarang.
- **Reward hacking.** Self-play membuat kedua belah pihak lebih pintar, tetapi hanya pada permainan *sebagaimana Anda mendefinisikannya*. Jika sistem imbalan Anda memiliki celah yang tidak disengaja (seperti memberi imbalan kepada pemain hanya karena bertahan lebih lama daripada menang), kedua belah pihak akan saling mengeksploitasi celah tersebut secara bersama-sama, yang mengarah pada perilaku yang aneh dan tidak membantu alih-alih menguasai permainan yang sebenarnya.

---

## Kata Kunci untuk Diingat

| Kata | Makna |
|------|---------|
| **Self-play**      | Agen yang sama memainkan kedua sisi permainan |
| **Zero-sum**       | Keuntungan satu pemain = kerugian pemain lain |
| **Simetri**       | Satu tabel-Q dapat melayani kedua belah pihak jika Anda membalik tanda |
| **Population play**| Self-play dengan *banyak* versi masa lalu Anda sendiri sebagai lawan (AlphaStar) |
| **Kurikulum**     | Progresi kesulitan alami — self-play mendapatkannya secara gratis |
| **MCTS**           | Monte-Carlo Tree Search — algoritma perencanaan yang dipasangkan AlphaZero dengan self-play |

---

## Ringkasan Satu Kalimat

> **Self-play mengubah peningkatan menjadi tangganya sendiri: setiap kali Anda menjadi lebih baik, lawan Anda juga ikut meningkat — secara otomatis.**

Ide ini, yang ditingkatkan dengan **jaringan saraf** (fungsi matematika yang terinspirasi otak yang mempelajari pola dari data) dan pencarian pohon (tree search), telah mengalahkan manusia terbaik dalam Go, catur, shogi, Dota 2, dan StarCraft.
