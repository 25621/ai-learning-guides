# DPO: Melewati Sang Hakim dan Langsung ke Sumbernya

## Ide Besar

RLHF klasik memiliki dua tahap: pertama latih model imbalan, lalu gunakan PPO untuk mengejar skornya. DPO (Direct Preference Optimization) mengajukan pertanyaan cerdas:

*Jika model imbalan hanyalah langkah perantara, bisakah kita melewatinya?*

Ternyata: ya. DPO melatih model bahasa secara langsung dari pasangan preferensi, tanpa hakim terpisah, tanpa loop pengambilan sampel PPO, dan tanpa koefisien KL untuk disetel. Ia menggunakan satu formula yang elegan dan berperilaku seperti pembelajaran terawasi (supervised learning).

Ini membuat DPO lebih sederhana untuk dijalankan, lebih stabil, dan lebih cepat - itulah sebabnya DPO dengan cepat menjadi pilihan utama untuk banyak model sumber terbuka yang selaras (aligned models).

## Analogi Kehidupan Nyata

Misalkan Anda sedang melatih seorang siswa untuk menulis esai.

Pendekatan PPO adalah: sewa seorang guru untuk menilai esai, lalu minta siswa menulis esai demi esai dan menyesuaikannya berdasarkan nilai dari guru.

Pendekatan DPO adalah: tunjukkan kepada siswa dua esai sekaligus dan katakan, "yang ini lebih baik - cenderunglah menulis seperti yang ini, dan menjauhlah dari yang itu." Tidak ada guru di tengahnya. Siswa menyesuaikan diri secara langsung dari perbandingan.

Keduanya bisa berhasil. DPO biasanya selesai lebih cepat karena tidak ada yang harus melatih dan memelihara guru yang terpisah.

## Cara Kerjanya (Intuisi Saja)

DPO menggunakan pasangan preferensi yang sama dengan pemodelan imbalan: prompt, pilihan (chosen), ditolak (rejected). Untuk setiap pasangan, ia mengajukan dua pertanyaan:

1. Apakah model menjadi **lebih mungkin** menghasilkan respons pilihan daripada model referensi?
2. Apakah model menjadi **kurang mungkin** menghasilkan respons yang ditolak daripada model referensi?

Pelatihan mendorong kedua angka tersebut ke arah yang benar sekaligus. Yang krusial, model referensi selalu ada dalam perbandingan - ia memainkan peran yang sama dengan penalti KL di PPO. Model diperbolehkan untuk bergeser, tetapi pergeseran tersebut selalu *relatif terhadap* titik awal.

Hasil halus dan indah dari makalah DPO adalah bahwa fungsi loss tunggal ini secara matematis setara dengan "latih model imbalan, lalu jalankan PPO dengan penalti KL." Tujuan yang sama, perjalanan yang lebih sederhana.

## Apa yang Ditunjukkan Eksperimen

Kami melatih sebuah kebijakan secara langsung pada 2.000 pasangan preferensi selama 300 epoch.

![Pelatihan DPO](outputs/dpo_implementation.png)

- **Kiri** - loss DPO turun saat model belajar untuk lebih menyukai respons pilihan daripada yang ditolak.
- **Tengah** - akurasi preferensi (seberapa sering kebijakan memberikan imbalan implisit yang lebih tinggi pada respons pilihan) naik hingga sekitar 99%.
- **Kanan** - margin imbalan implisit tumbuh. DPO tidak pernah menyebutkan "imbalan" secara eksplisit, tetapi selisih antara log-probabilitas pilihan vs ditolak, yang diskalakan oleh beta, dapat dibaca sebagai imbalan. Selisih ini melebar terus, artinya model menjadi lebih percaya diri dalam preferensinya.

Perhatikan betapa bersih tampilannya dibandingkan dengan PPO. Tidak ada loop pengambilan sampel, tidak ada noise eksplorasi, dan tidak ada model imbalan terpisah yang berjalan. Setiap epoch adalah pembaruan murni gaya terawasi (supervised-style) atas dataset preferensi.

## Di Mana Posisi Ini Dalam Pipeline RLHF

DPO adalah *alternatif* untuk langkah kedua dari pipeline klasik:

- **Klasik:** preferensi → model imbalan → PPO → model selaras.
- **DPO:** preferensi → model selaras. (Selesai.)

Masalahnya adalah DPO melatih pada dataset preferensi yang tetap. PPO, karena mengambil sampel respons baru di setiap putaran, secara prinsip dapat mengeksplorasi lebih jauh. Dalam praktiknya, DPO menang untuk sebagian besar kasus penggunaan "menyelaraskan pada dataset preferensi yang dikurasi".

## Mengapa Ini Penting di Luar Lab

Pola "melewati pengukuran menengah" ada di mana-mana:

- Seorang pelatih yang mengoreksi gerakan perenang dengan mendemonstrasikannya secara berdampingan daripada menilai setiap putaran dengan stopwatch.
- Seorang fotografer yang mengedit dua foto sekaligus, memilih yang lebih baik, daripada membangun rubrik penilaian "foto yang bagus".
- Seorang manajer perekrutan yang membandingkan dua resume daripada menilai masing-masing terhadap daftar periksa 30 poin.

Saat Anda hanya perlu *memperingkat (rank)*, Anda tidak memerlukan skala absolut. DPO adalah wawasan yang diterapkan pada model bahasa.

## Ringkasan Satu Kalimat

**DPO mengubah pasangan preferensi secara langsung menjadi model yang lebih baik, tanpa model imbalan di tengahnya - lebih sederhana dari PPO, dan sering kali sama bagusnya.**
