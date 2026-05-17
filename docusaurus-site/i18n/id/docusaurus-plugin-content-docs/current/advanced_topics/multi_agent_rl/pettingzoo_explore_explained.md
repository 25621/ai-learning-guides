# Menjelajahi Lingkungan PettingZoo 🦓

## Apa Itu PettingZoo?

Jika Anda pernah melakukan RL agen-tunggal, Anda mungkin pernah menggunakan **Gymnasium** (penerus OpenAI Gym). Setiap lingkungan terlihat sama: `env.reset()`, `env.step(action) → obs, reward, done, info` — sebuah *observasi* baru tentang dunia, sinyal *imbalan* skalar, flag *selesai* yang mengatakan "game over", dan kamus *info* untuk tambahan debugging. Keseragaman itulah yang membuat pustaka RL berfungsi.

**PettingZoo** adalah ide yang persis sama tetapi untuk *beberapa agen*. Ini adalah kebun binatang (zoo) dari lingkungan multi-agen — semuanya di balik satu API yang terdefinisi dengan baik:
- **Masalah mainan klasik**: lingkungan sederhana seperti Batu-Kertas-Gunting untuk menguji algoritma dasar.
- **Grid world kooperatif**: agen menavigasi kisi untuk mencapai tujuan bersama.
- **Multipemain Atari**: game kompetitif klasik seperti Pong.
- **MPE (Multi-Particle Environment)**: lingkungan fisika ruang-kontinu untuk koordinasi dan kompetisi yang kompleks.

Jika Anda dapat menulis kode yang berfungsi di satu lingkungan PettingZoo, Anda dapat mencolokkannya ke lingkungan lainnya hampir tanpa perubahan.

---

## Dua Gaya API

Pengaturan multi-agen lebih berantakan daripada agen-tunggal karena dua agen dapat bertindak pada saat yang sama, atau bergantian, atau bahkan dalam urutan sewenang-wenang. PettingZoo menyelesaikan ini dengan dua API paralel:

### 1) AEC (Agent-Environment-Cycle)

Satu agen bertindak pada satu waktu. Lingkungan melakukan loop melalui agen dalam beberapa urutan, dan masing-masing mendapatkan:
- sebuah **observasi** — apa yang mereka lihat *sekarang*,
- sebuah **imbalan** — imbalan yang diperoleh dari tindakan *bersama* di putaran penuh terakhir (yaitu, apa yang terjadi sebagai hasil dari tindakan *semua* agen, bukan hanya Anda; dalam permainan catur, misalnya, imbalan Anda mencerminkan status papan setelah langkah terakhir lawan, bukan hanya langkah Anda),
- sebuah **flag terminasi (termination flag)** — `True` ketika episode berakhir secara *alami* (misalnya, skakmat, seseorang menang),
- sebuah **flag pemotongan (truncation flag)** — `True` ketika episode *dipotong* oleh batas waktu sebelum akhir alami tercapai.

Ini alami untuk **permainan berbasis giliran (turn-based games)** seperti catur, Go, poker.

```python
env.reset()
for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    if term or trunc:
        env.step(None)
        continue
    action = my_policy(obs, agent)
    env.step(action)
```

### 2) Paralel

Semua agen mengamati dan bertindak secara simultan di setiap langkah. `step()` menerima *dictionary* tindakan dan mengembalikan dictionary observasi dan imbalan.

Ini alami untuk **permainan real-time** seperti MPE (Multi-Particle Environments, di mana semua agen titik bergerak secara simultan) atau gridworld multi-agen.

```python
obs, info = env.reset()
while env.agents:
    actions = {a: my_policy(obs[a]) for a in env.agents}
    obs, rewards, terms, truncs, info = env.step(actions)
```

Kedua gaya tersebut adalah **isomorfik** — setara secara struktural dan dapat dikonversi satu sama lain: lingkungan AEC apa pun dapat secara otomatis dibungkus agar terlihat seperti lingkungan Paralel, dan sebaliknya. PettingZoo menyediakan pembungkus konversi tersebut sehingga Anda hanya perlu menulis kode untuk satu gaya saja.

---

## Analogi Kehidupan Nyata

- **AEC = malam permainan papan.** "Giliran Alice. Sekarang Bob. Sekarang Carol. Kembali ke Alice." Siapa pun yang bergerak selanjutnya melihat status papan terbaru.
- **Paralel = video game multipemain.** Keempat pemain menekan tombol secara simultan; game memperbarui dunia 60 kali per detik.
- **Mengapa API seragam itu penting.** Bayangkan jika setiap video game multipemain membutuhkan joystick-nya sendiri. PettingZoo adalah "joystick universal" dari MARL.

---

## Apa yang Dilakukan Kode Kami

Kami membangun lingkungan **gaya PettingZoo** dari nol: **Iterated Coordination Game**. Dua agen berulang kali memilih saluran `0` atau `1`:

- Pilihan sama → keduanya mendapat +1
- Pilihan berbeda → keduanya mendapat -1

**Observasi** yang diterima setiap agen adalah *tindakan bersama* sebelumnya — apa yang dipilih kedua agen pada putaran terakhir, dikemas menjadi satu integer tunggal. Secara konkret: tindakan terakhir masing-masing agen adalah salah satu dari `{start, 0, 1}` (3 status), sehingga pasangannya dikodekan sebagai `3 × status_agen_1 + status_agen_2`, menghasilkan 9 integer yang mungkin (0 – 8). Integer 0 adalah status "start" — ini menandakan bahwa belum ada tindakan yang diambil (awal mula episode). Sebuah episode berlangsung selama 25 langkah, jadi total return maksimum adalah +25 per agen dan minimum adalah -25. **Permainan acak menghasilkan skor ≈ 0** karena pada setiap langkah dua agen acak independen masing-masing memilih 0 atau 1 dengan probabilitas yang sama: mereka cocok 50% dari waktu (+1) dan berbeda 50% dari waktu (-1), memberikan imbalan per langkah yang diharapkan sebesar 0,5 × (+1) + 0,5 × (-1) = **0**. Dijumlahkan selama 25 langkah, return episode yang diharapkan juga 0.

Kami kemudian:

1. **Mendemonstrasikan antarmuka AEC** dengan rollout acak — ini mengonfirmasi loop AEC dasar: `agent_iter()` menghasilkan agen yang gilirannya tiba, `last()` membaca observasi saat ini dan imbalan yang terakumulasi dari agen tersebut, dan `step()` memberikan tindakan yang mereka pilih kembali ke lingkungan.
2. **Melatih dua pembelajar-Q independen melalui antarmuka Paralel**. Masing-masing agen menyimpan tabel-Q-nya sendiri dengan kunci **observasi tindakan-bersama** (integer tunggal yang mengodekan apa yang dilakukan *kedua* agen pada putaran terakhir), sehingga ia dapat mempelajari "ketika kita berdua memilih 0 terakhir kali, saya harus memilih 0 lagi."
3. **Mencoba mengimpor pustaka `pettingzoo` yang asli** dan menjalankan salah satu lingkungan bawaannya (Batu-Kertas-Gunting) dengan kebijakan acak. Jika PettingZoo tidak terinstal, kita melewati langkah ini dengan pesan yang ramah.

### Apa yang seharusnya Anda lihat

| Tahap | Yang Diharapkan |
|-------|----------|
| Rollout acak (AEC)            | Rata-rata return episode di dekat **0** — agen acak memilih saluran secara independen, cocok dan tidak cocok dalam jumlah yang kira-kira sama. |
| Pembelajar-Q Independen (Paralel) — 100 episode pertama | Sekitar **0** — masih sebagian besar acak saat agen bereksplorasi. |
| Pembelajar-Q Independen — 100 episode terakhir             | Sangat positif, **+20 hingga +25** — **koordinasi telah muncul**: kedua agen telah belajar untuk memilih saluran yang sama secara andal di setiap putaran. |

Plot `outputs/pettingzoo_coordination.png` menunjukkan return episode individual (abu-abu) dan kurva **Rata-rata** bergulir (biru). Rata-rata menghaluskan episode yang berisik sehingga Anda dapat melihat trennya: agen berpindah dari permainan acak yang tidak terkoordinasi di dekat ~0 menuju **koordinasi** yang stabil di dekat ~+25. Garis hijau putus-putap menandai batas koordinasi sempurna.

Jika `pettingzoo` terinstal, skrip juga menjalankan `pettingzoo.classic.rps_v2` untuk membuktikan bahwa skrip berfungsi terhadap pustaka aslinya dengan cara yang sama seperti yang ia lakukan pada lingkungan buatan kita sendiri. Untuk mengaktifkan bagian tersebut:

```bash
source ../../venv/bin/activate
pip install "pettingzoo[classic]"
```

---

## Mengapa Membangun Lingkungan Kustom Terlebih Dahulu?

Karena **API adalah pelajarannya.** (Memahami cara menyusun interaksi antara beberapa agen dan lingkungan lebih penting daripada aturan permainan spesifiknya.) RL multi-agen memiliki banyak variasi (berbasis giliran, real-time, kooperatif, kompetitif, campuran), dan semuanya masuk ke dalam pola AEC / Paralel. Setelah Anda mengimplementasikan kedua loop tersebut, setiap lingkungan PettingZoo hanyalah masalah memasang konstruktor lingkungan yang berbeda — kode pelatihannya tetap sama.

Inilah tepatnya bagaimana Gymnasium mengubah RL agen-tunggal: dengan membuat lingkungan menjadi kotak hitam di balik antarmuka yang seragam.

---

## Di Mana Q-learning Independen Membantu dan Merugikan

Permainan koordinasi bersifat *pemaaf* — agen berbagi tanda imbalan, sehingga kepentingan mereka selaras. Pembelajar independen dapat menyelesaikan ini dengan senang hati karena setiap peningkatan oleh satu agen membantu agen lainnya.

Dalam permainan **adversarial (lawan)** (Batu-Kertas-Gunting) Q-learning independen berosilasi selamanya (saat satu agen beradaptasi, agen lainnya mengubah strateginya untuk membalas, menyebabkan pengejaran tanpa akhir). Dalam permainan **yang dapat diobservasi sebagian (partially-observable)** ia tidak dapat belajar sama sekali karena "observasi" hanyalah sepotong status (seorang agen mungkin dihukum karena tindakan yang baik hanya karena ia tidak dapat melihat apa yang dilakukan agen lain). PettingZoo mencakup kedua jenis lingkungan tersebut sehingga Anda dapat melihat mode kegagalan ini sendiri.

---

## Kata Kunci untuk Diingat

| Kata | Makna |
|------|---------|
| **PettingZoo**     | Gymnasium-nya RL multi-agen — pustaka lingkungan MARL standar |
| **AEC**            | Agent-Environment-Cycle: satu agen bertindak per langkah (berbasis giliran) |
| **Parallel API**   | Semua agen bertindak secara simultan di setiap langkah |
| **MPE**            | Multi-Particle Environment, testbed kooperatif/kompetitif populer yang disertakan dengan PettingZoo (sering kali melibatkan titik-titik bergerak yang menavigasi tugas berbasis fisika). |
| **CTDE**           | Centralised Training, Decentralised Execution — berlatih dengan pandangan global (akses ke semua status), dijalankan hanya dengan observasi lokal (masing-masing agen bertindak berdasarkan visinya sendiri yang terbatas). |
| **Independent Q-learning** | Masing-masing agen menjalankan Q-learning vanilla (algoritma Q-learning standar yang tidak dimodifikasi), mengabaikan keberadaan pembelajar lain. |

---

## Ringkasan Satu Kalimat

> **PettingZoo memberikan setiap lingkungan multi-agen bentuk yang sama — sehingga kode yang Anda tulis hari ini masih berfungsi besok pada permainan yang sama sekali berbeda.**

Setelah kedua gaya API tersebut menjadi kebiasaan, Anda dapat melangkah ke MADDPG (kritikus terpusat untuk agen kontrol-kontinu), QMIX (pencampuran nilai untuk tim kooperatif), MAPPO (PPO multi-agen), atau algoritma MARL modern lainnya — sisi lingkungan dari kode Anda tidak perlu berubah.
