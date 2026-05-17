# Tugas Cakrawala-Panjang (Long-Horizon Tasks)

## Ide Besar: Ketika Imbalan Masih Sangat Jauh {#the-big-idea-when-the-reward-is-very-far-away}

Bayangkan Anda adalah seorang koki yang mencoba mempelajari resep baru murni dengan mencicipi hidangan akhirnya. Anda mengikuti 40 langkah — potong, tumis, bumbui, rebus, sajikan — tetapi Anda hanya mendapatkan umpan balik di bagian paling akhir: "Terlalu asin." Langkah mana dari 40 langkah tersebut yang menyebabkan masalah? Anda tidak tahu.

Ini adalah **masalah cakrawala-panjang (long-horizon problem)**: ketika sinyal imbalan terpisah dari keputusan yang menyebabkannya oleh lusinan (atau ratusan) langkah, pembelajaran menjadi sangat sulit.

---

## Mengapa Agen "Datar" Kesulitan {#why-flat-agents-struggle}

Agen RL datar (seperti agen DQN dari Fase 3) mencoba mempelajari nilai dari setiap langkah sekaligus. Dalam tugas-tugas singkat — menyeimbangkan tiang, menghindari dinding — ini berfungsi dengan baik. Imbalan tiba dengan cepat, dan agen dapat menghubungkan sebab dan akibat.

Tetapi dalam tugas yang panjang — mengambil kunci, lalu menggunakannya untuk membuka pintu, lalu keluar dari labirin — agen harus:

1. Menemukan kunci secara tidak sengaja (beruntung!)
2. Mengingat bahwa mengambil kunci itu berguna
3. Menemukan pintu secara tidak sengaja (beruntung lagi!)
4. Menghubungkan seluruh urutan ke imbalan tunggal di pintu keluar

Dengan eksplorasi acak, peluang untuk menyelesaikan seluruh urutan ini secara tidak sengaja menyusut secara eksponensial dengan setiap langkah baru yang diperlukan. DQN datar pada dasarnya perlu beruntung berkali-kali sebelum melihat satu imbalan positif untuk dipelajari.

---

## Solusi Hierarkis: Pecah dan Taklukkan {#the-hierarchical-solution-divide-and-conquer}

RL Hierarkis memecah tugas panjang menjadi **struktur dua tingkat**:

| Tingkat | Disebut | Tugas |
|-------|--------|-----|
| Tinggi | **Manajer** | Memilih sub-tujuan berikutnya |
| Rendah | **Pekerja** | Menavigasi ke sub-tujuan tersebut |

Inilah tepatnya cara manusia menangani tugas-tugas kompleks. Anda tidak merencanakan perjalanan darat Anda selangkah demi selangkah sebelum berangkat. Sebaliknya:

- **Manajer (Anda, di rumah):** "Pemberhentian pertama: pom bensin. Pemberhentian berikutnya: pintu masuk jalan tol. Lalu: pintu keluar 42."
- **Pekerja (Anda, mengemudi):** Menangani semua keputusan penyetiran individu untuk mencapai setiap pemberhentian.

Manajer berpikir dalam hal *titik pemeriksaan (checkpoint)*. Pekerja berpikir dalam hal *roda setir*.

---

## Mengapa Ini Mengalahkan Pembelajaran Datar pada Tugas Panjang {#why-this-beats-flat-learning-on-long-tasks}

Pekerja hanya perlu mencapai *sub-tujuan berikutnya* — tugas singkat dengan imbalan yang jelas dan dekat. Ia mendapatkan umpan balik dengan cepat dan belajar secara efisien.

Manajer hanya perlu memutuskan *urutan sub-tujuan* — masalah yang jauh lebih sederhana daripada merencanakan setiap langkah individu.

Bersama-sama, kedua tingkat tersebut membagi masalah cakrawala-panjang yang sulit menjadi dua masalah cakrawala-pendek yang mudah.

---

## Eksperimen Kisi Kunci-Pintu {#the-key-door-grid-experiment}

Skrip kami menguji kedua pendekatan tersebut pada **kisi terbuka 9x9** dengan dua objek:

- Sebuah **KUNCI** di satu sudut (harus diambil terlebih dahulu).
- Sebuah **PINTU** di sudut yang berlawanan (hanya dihitung jika Anda memiliki kunci).

Satu-satunya imbalan nyata adalah +1 ketika agen mencapai pintu *setelah* mengambil kunci. Imbalan tunggal tersebut mengharuskan dua sub-tugas berurutan dirangkai dengan benar.

Dua agen bersaing:

**DQN Datar:** Harus menemukan kedua sub-tugas dalam urutan yang benar secara tidak sengaja, lalu merambatkan sinyal balik melalui keduanya. Karena keberhasilan membutuhkan dua temuan beruntung dalam satu episode, DQN jarang mempelajari sesuatu yang berguna.

**Agen Hierarkis:**
- Aturan manajer: "Pergi ke kunci dulu, lalu pergi ke pintu."
- Pekerja mendapatkan **+1 setiap kali mencapai sub-tujuan** — baik itu kunci atau pintu.
- Dua tugas singkat terpisah, masing-masing dengan imbalan terdekat yang jelas.

---

## Apa yang Ditunjukkan Grafik {#what-the-charts-show}

![Hasil Tugas Cakrawala-Panjang](outputs/long_horizon_tasks.png)

**Kiri — Tingkat Keberhasilan Seiring Waktu:** Agen hierarkis (biru) belajar memecahkan labirin jauh lebih awal daripada DQN datar (merah). Agen datar mungkin akhirnya belajar juga — dengan episode yang cukup — tetapi agen hierarkis sampai di sana lebih cepat karena sinyal pembelajarannya padat dan lokal.

**Kanan — Performa Akhir:** Diagram batang menunjukkan tingkat keberhasilan rata-rata selama 500 episode terakhir. Keunggulan agen hierarkis sudah jelas: memecah masalah menjadi sub-tujuan membuatnya dapat dikelola.

---

## Di Mana Pemikiran Cakrawala-Panjang Muncul {#where-long-horizon-thinking-shows-up}

| Domain | Contoh cakrawala-panjang |
|--------|---------------------|
| Robotika | Merakit perangkat dengan 30 bagian secara berurutan |
| Game | Memenangkan pertandingan catur (banyak langkah, satu pemenang) |
| Bahasa | Menulis makalah penelitian lengkap (banyak keputusan penulisan, satu skor kualitas) |
| Sains | Menjalankan eksperimen multi-bulan dan mengevaluasi hasil |

Inilah tepatnya mengapa Feudal Networks (arsitektur di mana manajer menetapkan tujuan arah untuk pekerja tingkat bawah) dan HIRO (Hierarchical RL with subgoals) ditemukan — karena RL datar membentur dinding pada masalah-masalah ini, dekomposisi hierarkis menjadi strategi dominan.

---

## Hubungan dengan Kebijakan Terkondisi-Tujuan {#the-connection-to-goal-conditioned-policies}

Perhatikan bahwa **pekerja** dalam agen hierarkis kami pada dasarnya adalah sebuah **kebijakan terkondisi-tujuan** — ia menerima sub-tujuan dan menavigasi ke arahnya. Ini adalah desain standar dalam HIRO dan makalah terkait: manajer menetapkan tujuan, pekerja adalah kebijakan terkondisi-tujuan yang mengejarnya.

Kedua ide tersebut — kebijakan terkondisi-tujuan dan struktur hierarkis — adalah dua sisi dari koin yang sama, itulah sebabnya mereka muncul bersama dalam modul ini.

---

## Ringkasan Satu Kalimat {#one-sentence-summary}

> **Tugas cakrawala-panjang itu sulit karena imbalan datang terlambat untuk mengajarkan keputusan individu — RL hierarkis menyelesaikan ini dengan menyisipkan sub-tujuan terdekat yang membiarkan pekerja belajar dengan cepat sementara manajer menangani urutan gambaran besarnya.**
