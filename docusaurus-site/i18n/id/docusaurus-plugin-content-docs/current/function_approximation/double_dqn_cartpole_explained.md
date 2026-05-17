# Double DQN: Memperbaiki Masalah Kepercayaan Diri Berlebih 🤔

## Masalah: DQN Mengira Dirinya Lebih Baik Daripada yang Sebenarnya

Bayangkan Anda ditanya: "Apa restoran terbaik di kota ini?"

Anda mungkin berkata "Pizza Palace luar biasa — itu pasti 10/10!" Tapi Anda baru ke sana dua kali. Anda tidak benar-benar tahu apakah itu *benar-benar* 10/10. Anda mungkin melebih-lebihkan karena Anda beruntung mendapatkan pizza yang enak pada dua kunjungan tersebut.

Masalah yang sama terjadi pada DQN: agen **melebih-lebihkan (overestimate) nilai-Q**.

---

## Mengapa DQN Melebih-lebihkan?

Saat DQN menghitung target, ia melakukan:
> target = imbalan + γ × **max** Q(status_berikutnya)

Bagian `max` itulah masalahnya! Saat Anda memilih maksimum dari beberapa perkiraan yang berisik (noisy), Anda hampir selalu memilih satu dengan kesalahan acak terbesar (bias ke atas).

**Contoh kehidupan nyata:** Anda meminta 5 teman menebak tinggi sebuah gedung. Tebakan mereka adalah: 40m, 38m, 45m (tebakan beruntung!), 39m, 41m. Tinggi aslinya adalah 40m. Jika Anda menggunakan `max(tebakan)` = 45m, Anda salah besar! Maksimum dari tebakan yang berisik hampir selalu merupakan perkiraan yang berlebihan.

Selama ribuan pembaruan, DQN terus berlatih menuju target yang terlalu tinggi ini, mempelajari bahwa segala sesuatunya lebih baik daripada yang sebenarnya. Ini dapat memperlambat pembelajaran atau menyebabkan agen membuat keputusan yang terlalu percaya diri dan buruk.

---

## Solusi Double DQN

**Double DQN** (Hasselt et al., 2016) membagi `max` menjadi dua langkah:

**Langkah 1 — Tindakan mana?** Gunakan **online network** untuk memilih tindakan terbaik:
> tindakan_terbaik = argmax Q_online(status_berikutnya)

**Langkah 2 — Berapa nilainya?** Gunakan **target network** untuk mengevaluasi tindakan tersebut:
> target = imbalan + γ × Q_target(status_berikutnya, tindakan_terbaik)

```
DQN Biasa:     target = r + γ × max_a Q_target(s', a)
                                 ↑ jaringan yang sama memilih DAN mengevaluasi → bias

Double DQN:    aksi_terbaik = argmax_a Q_online(s', a)   ← online yang memilih
               target = r + γ × Q_target(s', aksi_terbaik) ← target yang mengevaluasi
                                 ↑ jaringan berbeda → bias berkurang
```

**Contoh kehidupan nyata:** Dalam wawancara kerja, Anda tidak membiarkan pelamar kerja menilai sendiri tes kinerjanya (itulah masalah DQN biasa!). Sebaliknya, kandidat *mengajukan* karya terbaik mereka, dan penguji *terpisah* mengevaluasinya. Dua orang berbeda = penilaian yang lebih adil!

---

## Mengapa Pemisahan Ini Membantu?

Kedua jaringan (online dan target) memiliki bobot yang berbeda karena target jarang diperbarui. Mereka memiliki "pendapat" yang berbeda tentang tindakan mana yang terbaik.

Saat mereka tidak setuju:
- Online berkata: "Tindakan A terlihat bagus!"
- Target berkata: "Sebenarnya, Tindakan A biasa saja — nilainya sekitar 7, bukan 10"

Dengan menggunakan perkiraan NILAI jaringan target untuk tindakan yang DIPILIH jaringan online, kita mendapatkan angka yang lebih jujur dan tidak terlalu tinggi.

---

## Perbedaan Kode: Hanya Satu Baris!

Satu-satunya perubahan kode dari DQN biasa ke double DQN adalah dalam perhitungan target:

```python
# DQN Biasa:
q_next = target_net(s_next).max(dim=1).values

# Double DQN:
best_actions = online_net(s_next).argmax(dim=1, keepdim=True)   # pilih dengan online
q_next = target_net(s_next).gather(1, best_actions)              # evaluasi dengan target
```

Hanya dua baris yang berubah — tetapi dampaknya pada stabilitas dan akurasi sangat signifikan!

---

## Apa yang Ditunjukkan Perbandingannya

Saat Anda menjalankan `double_dqn_cartpole.py`, Anda akan melihat dua plot:

**Plot 1: Kurva Pembelajaran**
- Baik DQN biasa maupun double DQN harusnya bisa menyelesaikan CartPole
- Double DQN seringkali konvergen lebih cepat dan lebih stabil
- CartPole cukup sederhana sehingga perbedaannya tidak terlalu besar; perbedaannya lebih dramatis pada Atari

**Plot 2: Estimasi Nilai-Q**
- DQN Biasa: nilai-Q bergeser ke atas seiring waktu (overestimation)
- Double DQN: nilai-Q tetap lebih rendah dan akurat

Plot overestimasi nilai-Q adalah wawasan kuncinya — ini menunjukkan DQN biasa mempelajari nilai-nilai tinggi yang akhirnya merusak performa.

---

## Keluarga Peningkatan DQN

Double DQN hanyalah salah satu dari banyak peningkatan pada DQN biasa. Makalah "Rainbow" (2017) menggabungkan 6 peningkatan:

1. **Double DQN** (memperbaiki overestimasi) ← skrip ini!
2. **Prioritized Replay** (belajar lebih banyak dari pengalaman yang mengejutkan)
3. **Dueling Networks** (memisahkan "seberapa baik status ini?" dari "apa tindakan terbaiknya?")
4. **Multi-step returns** (melihat lebih jauh ke masa depan)
5. **Distributional RL** (mempelajari distribusi penuh dari return, bukan hanya rata-ratanya)
6. **NoisyNets** (eksplorasi yang dipelajari alih-alih [ε-greedy](../foundations/multi_armed_bandit_explained.md#the-epsilon-greedy-strategy))

Rainbow menggabungkan SEMUANYA dan mencapai performa Atari terbaik pada masanya!

---

## Kosakata Kunci

| Kata | Makna |
|------|---------|
| **Overestimation** | Nilai-Q lebih tinggi dari nilai sebenarnya (terlalu optimis) |
| **Double DQN** | Menggunakan jaringan online untuk pemilihan tindakan, jaringan target untuk evaluasi |
| **Decoupling** | Memisahkan dua tugas yang sebelumnya dilakukan oleh jaringan yang sama |
| **Bias** | Kesalahan sistematis dalam satu arah (selalu terlalu tinggi, atau selalu terlalu rendah) |
| **Rainbow** | Varian DQN yang menggabungkan 6 peningkatan untuk performa maksimal |

---

## Ringkasan Perjalanan Fase 3

Anda sekarang telah menyelesaikan progresi penuh Fase 3:

| Algoritma | Apa yang ditambahkan | Mengapa ini membantu |
|-----------|-------------|-------------|
| Linear Q | Jaringan saraf → rumus sederhana | Menangani status kontinu |
| Naive DQN | Jaringan saraf lengkap | Mempelajari pola yang kompleks |
| + Replay buffer | Pengambilan sampel memori acak | Memutus korelasi |
| + Target network | Salinan beku untuk target | Menstabilkan "titik bidik" |
| Atari DQN | CNN + penumpukan bingkai | Belajar dari piksel! |
| Double DQN | Pemisahan pilih/evaluasi | Mengurangi overestimasi |

Setiap langkah memecahkan masalah tertentu. Begitulah cara kerja riset yang sebenarnya — satu peningkatan hati-hati pada satu waktu!
