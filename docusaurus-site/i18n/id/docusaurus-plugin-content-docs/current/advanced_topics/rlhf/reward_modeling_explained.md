# Pemodelan Imbalan: Mengajari Komputer Apa yang Disukai Manusia

## Ide Besar

Model imbalan (reward model) adalah hakim kecil. Anda menunjukkan dua jawaban untuk pertanyaan yang sama, memberi tahu mana yang lebih disukai seseorang, dan seiring waktu ia belajar memberikan skor yang lebih tinggi untuk jawaban yang akan disukai orang.

Mengapa kita membutuhkan hakim seperti itu? Karena sebagian besar dari apa yang kita inginkan dari model bahasa sulit dituliskan sebagai rumus matematika. Tidak ada persamaan tunggal untuk "membantu", "sopan", atau "ditulis dengan baik". Tetapi orang hampir selalu dapat menunjukkan mana yang lebih baik dari dua pilihan. Model imbalan mengubah suara "yang ini lebih baik" yang sederhana menjadi skor yang dapat digunakan oleh algoritma pembelajaran.

## Analogi Kehidupan Nyata

Bayangkan mengajari teman cara membuat brownie.

Anda tidak memberikan buku aturan setebal 50 halaman tentang "apa yang membuat brownie yang enak". Sebaliknya, Anda mencicipi dua kelompok masakan dan berkata:

"Yang ini lebih enak."

Setelah beberapa putaran ini, teman Anda mulai memperhatikan polanya. Mungkin yang lebih lumer selalu menang. Mungkin yang terlalu matang selalu kalah. Teman Anda membangun sistem penilaian mental dari perbandingan Anda.

Model imbalan melakukan hal yang sama, tetapi dengan angka. Ia tidak perlu tahu *mengapa* jawaban yang dipilih lebih baik. Ia hanya membutuhkan banyak contoh "ini mengalahkan itu" dan secara bertahap ia mempelajari skor yang sejalan dengan preferensi tersebut.

## Cara Kerjanya (Intuisi Saja)

Setiap contoh adalah tiga bagian: sebuah prompt, respons **pilihan (chosen)**, dan respons **ditolak (rejected)**. Kita ingin model memberikan skor yang lebih tinggi pada yang dipilih daripada yang ditolak - dengan selisih berapa pun.

Dorongan pelatihannya sederhana dalam semangatnya:

- Skor pilihan terlalu rendah? Dorong ke atas.
- Skor yang ditolak terlalu tinggi? Dorong ke bawah.
- Sudah dalam urutan yang benar dengan celah yang jelas? Biarkan saja.

Dorongan itu disebut loss Bradley-Terry, dan merupakan resep standar dalam sistem RLHF modern.

## Apa yang Ditunjukkan Eksperimen

Kami melatih model imbalan pada 2.000 pasangan preferensi sintetis. Plot di bawah ini menunjukkan tiga pandangan dari sesi pelatihan yang sama.

![Pelatihan model imbalan](outputs/reward_modeling.png)

- **Kiri** - loss turun dengan cepat. Model menjadi lebih percaya diri tentang peringkatnya.
- **Tengah** - akurasi preferensi naik hingga hampir 100%. Pada hampir setiap pasangan, respons pilihan mendapat skor lebih tinggi daripada yang ditolak.
- **Kanan** - distribusi skor untuk respons pilihan vs ditolak saling menjauh. Pada awal pelatihan keduanya tumpang tindih; setelah pelatihan, respons pilihan duduk jelas di sebelah kanan.

Pemisahan itulah inti dari semuanya. Langkah selanjutnya (PPO atau DPO) sekarang dapat menggunakan skor ini sebagai target untuk dioptimalkan.

## Di Mana Posisi Ini Dalam Pipeline RLHF

Roadmap mendeskripsikan RLHF sebagai "menyelaraskan model dengan preferensi manusia". Model imbalan adalah langkah pertama dari tiga langkah:

1. **Model imbalan (file ini)** - mengubah suara preferensi menjadi skor.
2. **PPO fine-tuning** - mendorong model bahasa menuju skor yang lebih tinggi sambil tetap dekat dengan perilaku aslinya.
3. **DPO** - pintasan baru yang melewatkan model imbalan sepenuhnya.

Jadi pemodelan imbalan adalah jembatan antara *penilaian manusia* dan *optimasi mesin*. Salah membangun jembatan ini, maka setiap langkah hilir akan salah arah.

## Mengapa Ini Penting di Luar Lab

Ide yang sama muncul di banyak tempat:

- **Sistem rekomendasi** mempelajari apa yang Anda sukai dari klik, skip, dan waktu yang dihabiskan untuk menonton.
- **Mesin pencari** mempelajari peringkat dari "hasil mana yang Anda klik".
- **Restoran** mempelajari hidangan populer dari pesanan berulang, bukan dari pelanggan yang menulis esai tentang apa yang mereka sukai.

Kapan pun lebih mudah untuk *membandingkan* daripada *menilai*, model imbalan adalah alat yang tepat.

## Ringkasan Satu Kalimat

**Model imbalan adalah hakim terpelajar yang mengubah preferensi "yang ini lebih baik" menjadi skor numerik yang dapat dioptimalkan oleh sistem RLHF lainnya.**
