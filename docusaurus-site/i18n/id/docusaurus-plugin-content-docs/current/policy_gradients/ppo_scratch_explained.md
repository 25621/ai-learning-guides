# PPO: Pembaruan Kebijakan yang Aman dan Stabil

## Masalah dengan A2C

Bayangkan Anda sedang belajar menyeimbangkan gagang sapu di atas jari Anda. Setelah berminggu-minggu berlatih, Anda bisa menjaganya tetap tegak selama 30 detik!

Sekarang pelatih Anda memberi saran: "Miringkan pergelangan tangan Anda sedikit lebih ke kiri."

**Saran bagus → perubahan hati-hati → tetap seimbang selama 30 detik ✓**

Tetapi bagaimana jika pelatih bereaksi berlebihan dan berkata: "MIRINGKAN JAUH KE KIRI SEKARANG JUGA!" Anda mengoreksi secara berlebihan → gagang sapu jatuh → Anda kehilangan kemajuan berminggu-minggu.

Ini adalah masalah A2C: **pembaruan gradien yang besar dapat menghancurkan kebijakan yang baik**.

**PPO (Proximal Policy Optimization)** adalah sistem keamanan yang mencegah hal ini.

---

## Ide Utama: Tetap Dekat dengan Apa yang Berhasil

Batasan utama PPO:

> **"Jangan mengubah kebijakan terlalu banyak dalam satu pembaruan."**

Sebelum pembaruan, kita memiliki kebijakan "lama" π_old. Setelah pembaruan, kita memiliki kebijakan "baru" π_new.

PPO mengukur seberapa banyak kebijakan berubah dengan **rasio probabilitas**:

```
r(θ) = π_new(a|s) / π_old(a|s)
```

- r = 1,0: kebijakan tidak berubah
- r = 1,5: kebijakan baru 50% lebih mungkin mengambil tindakan tersebut
- r = 0,5: kebijakan baru 50% kurang mungkin mengambil tindakan tersebut

**Contoh kehidupan nyata:** Anda adalah seorang koki yang menyesuaikan resep.
- r = 1,0: jumlah garam yang sama seperti sebelumnya
- r = 2,0: dua kali lipat garam — terlalu ekstrem!
- r = 0,9: 10% lebih sedikit garam — perubahan kecil yang aman

---

## Trik Clipping {#the-clipping-trick}

PPO memotong (clip) rasio agar tetap berada di dalam [1-ε, 1+ε] (biasanya ε = 0,2):

```
L_CLIP = E[min(r(θ) · A,  clip(r(θ), 1-ε, 1+ε) · A)]
```

Mari kita bedah ini:

**Kasus 1: Tindakannya BAGUS (A > 0)**

Kita ingin melakukan tindakan ini lebih sering (r > 1). Tetapi kita membatasi seberapa banyak kita meningkatkannya:
```
jika r > 1,2: potong (clip) ke 1,2, tidak ada lagi insentif untuk mendorong lebih jauh
```
Ini mencegah kita berayun TERLALU jauh ke satu arah.

**Kasus 2: Tindakannya BURUK (A < 0)**

Kita ingin melakukan tindakan ini lebih jarang (r < 1). Tetapi sekali lagi, kita membatasi:
```
jika r < 0,8: potong (clip) ke 0,8, tidak ada lagi penalti untuk pergi lebih jauh
```

**Visual:**
```
ε = 0,2, jadi jendela rasio yang aman adalah 0,8 hingga 1,2.

Tindakan BAGUS (A > 0): tingkatkan probabilitas tindakan, tetapi berhenti menghargainya setelah 1,2
rasio r:       0,6      0,8      1,0      1,2      1,4
insentif:       ↑        ↑        ↑        ↑        -
makna:    terlalu rendah  ok      lama     maks   terpotong (clipped)

Tindakan BURUK (A < 0): kurangi probabilitas tindakan, tetapi berhenti menghargainya di bawah 0,8
rasio r:       0,6      0,8      1,0      1,2      1,4
insentif:       -        ↓        ↓        ↓        ↓
makna:      terpotong   maks     lama      ok   terlalu tinggi
```

Tanda `-` menandai wilayah datar yang terpotong. Di wilayah tersebut, membuat rasio probabilitas menjadi lebih ekstrem tidak meningkatkan tujuan, sehingga PPO tidak memiliki insentif tambahan untuk mendorong lebih jauh.

**Contoh kehidupan nyata:** Pembatas kecepatan mobil. Anda dapat berakselerasi, tetapi begitu Anda mencapai 120 km/jam, pembatas akan bekerja dan tidak membiarkan Anda melaju lebih cepat. Ini menjaga Anda tetap aman tanpa menghentikan Anda untuk bergerak.

---

## Mengapa Ini Mencegah Pembaruan Katastropik

Sebuah **pembaruan katastropik** adalah ketika satu perubahan kebijakan besar menghancurkan sepenuhnya semua yang telah dipelajari agen — pelatihan berjam-jam hilang dalam satu langkah gradien.

Tanpa clipping: satu langkah gradien besar dapat mengubah kebijakan secara drastis. Dengan clipping: gradiennya nol di luar [1-ε, 1+ε], sehingga kebijakan hanya bisa bergerak sedikit per langkah.

**Contoh kehidupan nyata:** Ahli bedah yang baik melakukan sayatan kecil dan presisi — bukan yang besar dan menyapu. PPO adalah "ahli bedah yang hati-hati" dari pengoptimal RL.

---

## GAE: Estimasi Advantage yang Lebih Pintar {#gae-smarter-advantage-estimates}

PPO menggunakan **Generalized Advantage Estimation (GAE)** untuk menghitung advantage:

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)          (TD error)
A_t = δ_t + γλ · δ_{t+1} + (γλ)² · δ_{t+2} + ...
```

GAE memiliki parameter λ (lambda):
- λ = 0: gunakan hanya one-step TD error (variansi rendah, bias tinggi)
- λ = 1: gunakan return Monte Carlo penuh (variansi tinggi, bias rendah)
- λ = 0,95: keseimbangan yang baik antara keduanya!

**Contoh kehidupan nyata:** Merencanakan perjalanan darat.
- λ=0: hanya lihat 5 mil ke depan (aman, tapi mungkin melewatkan jalan pintas nanti)
- λ=1: pertimbangkan seluruh perjalanan 500 mil (lebih banyak info, tapi sangat tidak pasti)
- λ=0,95: lihat jauh ke depan tetapi beri bobot lebih besar pada jalan-jalan terdekat ← keseimbangan terbaik!

---

## Multiple Epochs: Menggunakan Kembali Data Secara Efisien

Setelah mengumpulkan satu batch pengalaman (rollout), REINFORCE membuangnya setelah SATU kali pembaruan.

PPO menggunakan kembali setiap batch selama **K epoch** (biasanya 4-10 kali sapuan melalui data yang sama):

```
Kumpulkan 512 langkah × 4 lingkungan = 2048 transisi
Epoch 1: 32 minibatch × perbarui masing-masing
Epoch 2: acak, 32 minibatch lagi × perbarui masing-masing
Epoch 3: ...
Epoch 4: ...
```

**Apa itu "minibatch"?** Memperbarui dengan semua 2048 transisi sekaligus lambat dan lapar memori; memperbarui satu transisi pada satu waktu sangat berisik. **Minibatch** adalah potongan kecil di antaranya — di sini, 2048 ÷ 32 = **64 transisi per minibatch**. Kita menghitung satu langkah gradien per minibatch, sehingga setiap epoch melakukan 32 pembaruan kecil yang stabil alih-alih 1 yang besar. (Ini adalah ide minibatch yang sama yang digunakan di mana-mana dalam deep learning — lihat [mini-batch gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Mini-batch_gradient_descent).)

Clipping memastikan beberapa sapuan ini tidak melampaui target — tanpa clipping, multiple epoch akan menghancurkan kebijakan dengan mendorongnya terlalu jauh!

**Contoh kehidupan nyata:** Seorang siswa memiliki 30 soal latihan.
- REINFORCE: kerjakan setiap soal sekali, belajar sedikit, buang soalnya
- PPO: kerjakan setiap soal 4 kali (sudut pandang berbeda setiap kali), batasi perubahan Anda sehingga Anda tidak menghafal pola yang salah

---

## Loss Lengkap PPO

```
L = L_CLIP - c₁ · L_entropy + c₂ · L_critic

L_CLIP    = gradien kebijakan terpotong (clipped policy gradient)
L_entropy = bonus entropi (mendorong eksplorasi)
L_critic  = MSE antara V(s) dan return
```

Koefisien tipikal: c₁ = 0,01 (entropi), c₂ = 0,5 (kritikus)

**Dua istilah yang layak dibahas:**

- **Gradien kebijakan (Policy gradient)** — setengah bagian "aktor" dari loss. Ia menggunakan sinyal gradien untuk mendorong kebijakan ke arah tindakan dengan advantage yang lebih tinggi dan menjauh dari tindakan dengan advantage yang lebih rendah. Ini adalah ide inti yang sama yang diperkenalkan di REINFORCE — lihat [panduan REINFORCE](./reinforce_cartpole_explained.md#the-old-way-vs-the-new-way) untuk intuisinya. PPO hanya menambahkan pembungkus clipping di sekitarnya.
- **MSE (Mean Squared Error)** — setengah bagian "kritikus" dari loss. Kritikus V(s) memprediksi return yang diharapkan dari suatu status; kita membandingkan prediksinya dengan return aktual dan menguadratkan selisihnya: `MSE = mean((V(s) - return)²)`. Penguadratan menghukum kesalahan besar lebih berat daripada yang kecil dan memberikan sinyal yang halus dan dapat dideferensiasi untuk pelatihan. (Loss regresi standar — lihat [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error).)

---

## Hasil

```
Pembaruan 200 | Rata-rata imbalan: ~120
Pembaruan 400 | Rata-rata imbalan: ~280
Pembaruan 800 | Rata-rata imbalan: ~280-300
```

PPO di CartPole menunjukkan peningkatan yang stabil tetapi cenderung mendatar (plateau) di sekitar 280-300. (**Plateau** berarti kurva pembelajaran mendatar — imbalan berhenti membaik meskipun pelatihan berlanjut. Kebijakan telah menemukan strategi yang baik secara lokal tetapi tidak membuat kemajuan lebih lanjut.) Ini sebenarnya sudah diperkirakan — PPO dirancang untuk lingkungan yang lebih sulit dengan episode yang lebih panjang.

Observasi menarik: **REINFORCE menyelesaikan CartPole lebih cepat!** (rata-rata 500 vs rata-rata 300)

Mengapa? Episode CartPole pendek (≤500 langkah), sehingga return tepat REINFORCE sangat akurat. Estimasi bootstrap PPO menambahkan kompleksitas yang tidak perlu. PPO benar-benar bersinar pada lingkungan di mana menunggu episode penuh tidak praktis (seperti BipedalWalker).

**Apa itu "BipedalWalker"?** BipedalWalker (khususnya `BipedalWalker-v3` di [Gymnasium](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)) adalah lingkungan benchmark RL klasik: robot berkaki 2 yang harus belajar berjalan ke depan melintasi medan yang tidak rata tanpa jatuh. Berbeda dengan dua tindakan diskret CartPole (KIRI / KANAN), BipedalWalker memiliki tindakan **kontinu** — empat nilai torsi, satu untuk setiap sendi kaki, masing-masing berupa angka riil dalam [-1, 1]. Episode dapat berjalan selama ribuan langkah, yang merupakan ranah di mana efisiensi data dan stabilitas PPO membuahkan hasil.

---

## Persamaan Utama

```
Rasio:       r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
Clip loss:   L_CLIP = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) · A_t)]
GAE:         A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

---

## Poin-Poin Penting

| Konsep | Bahasa Sederhana |
|---------|---------------|
| **Rasio r(θ)** | Seberapa banyak kebijakan berubah pada tindakan ini |
| **Clip ε** | Batas keamanan — jangan mengubah kebijakan lebih dari ini |
| **GAE** | Cara cerdas untuk memperkirakan advantage dengan melihat beberapa langkah ke depan |
| **Efisiensi data** | Setiap rollout dikumpulkan dari beberapa lingkungan paralel (pengalaman yang stabil dan tidak berkorelasi) dan kemudian digunakan kembali untuk K epoch pembaruan minibatch — clipping menjaga sapuan berulang ini tetap aman |

---

## Apa Selanjutnya?

Sejauh ini, semua lingkungan kita memiliki tindakan **diskret** (dorong KIRI atau KANAN).

Robot sungguhan perlu mengontrol tindakan **kontinu** — seperti "berikan gaya tepat 0,73 Newton."

`ppo_continuous.py` memperluas PPO ke tindakan kontinu menggunakan **kebijakan Gaussian**, dan mengujinya pada lingkungan BipedalWalker-v3 yang jauh lebih sulit!
