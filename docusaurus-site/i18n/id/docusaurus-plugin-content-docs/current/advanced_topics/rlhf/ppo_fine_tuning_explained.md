# PPO Fine-Tuning: Menyempurnakan Model Tanpa Merusaknya

## Ide Besar

Setelah kita memiliki model imbalan yang menilai respons, kita ingin model bahasa kita menghasilkan respons dengan skor yang lebih tinggi. PPO (Proximal Policy Optimization) melakukan hal ini - tetapi ia menambahkan "sabuk pengaman" agar model tidak hanya mengejar skor dan lupa cara menulis teks yang normal.

Bayangkan ini sebagai langkah pemolesan. Model tersebut sudah berbicara dengan lancar; kita hanya mendorongnya untuk berbicara dengan cara yang dihargai oleh model imbalan, sambil tetap menjaga suaranya agar tetap dikenali.

## Analogi Kehidupan Nyata

Bayangkan seorang koki yang sudah memasak dengan baik tetapi sekarang sedang belajar untuk menyenangkan kritikus makanan tertentu.

Setelah setiap hidangan, kritikus memberikan skor. Koki memiliki dua tekanan:

1. **Dapatkan skor yang lebih tinggi.** Masak dengan cara yang disukai kritikus.
2. **Jangan menjadi tidak dikenali.** Jika koki meninggalkan gaya mereka sendiri sepenuhnya - memasukkan garam satu cangkir penuh hanya untuk mengejar skor - makanannya menjadi aneh. Pelanggan berhenti datang.

PPO menangkap kedua tekanan tersebut:

- Bagian **imbalan (reward)** mendorong model ke arah respons yang disukai hakim.
- Bagian **penalti KL** menarik model kembali ke cara berbicaranya sebelum pelatihan dimulai. KL hanyalah cara mengukur "seberapa berbeda perilaku baru dari perilaku lama."

Bersama-sama mereka berkata: *menjadi lebih baik, tetapi tetaplah menjadi diri sendiri*.

## Cara Kerjanya (Intuisi Saja)

Setiap putaran pelatihan terlihat seperti ini:

1. Ambil beberapa prompt. Biarkan model saat ini menghasilkan respons.
2. Beri skor respons dengan model imbalan.
3. Bandingkan dengan **model referensi** - salinan beku dari model sebelum pelatihan. Jika respons baru sangat berbeda, kurangi penalti KL dari imbalan.
4. Dorong model ke arah respons yang mendapat skor baik.

Kata "Proximal" dalam PPO berarti *jangan mengambil lompatan besar*. Setiap pembaruan adalah langkah kecil yang hati-hati. Lompatan besar dalam pelatihan kebijakan menyebabkan kegagalan, itulah sebabnya metode sebelumnya seperti vanilla policy gradient sangat tidak stabil.

## Apa yang Ditunjukkan Eksperimen

Kita mulai dengan kebijakan baru dan model imbalan yang terlatih. PPO berjalan selama 150 iterasi, mengambil sampel batch respons dan memperbarui kebijakan.

![Pelatihan PPO](outputs/ppo_fine_tuning.png)

- **Kiri** - skor rata-rata model imbalan naik terus. Kebijakan belajar menghasilkan respons yang disukai hakim.
- **Tengah** - divergensi KL dari model referensi tumbuh. Kebijakan menjauh dari titik awalnya. Ini diharapkan, tetapi jika tumbuh tanpa terkendali, model akan melayang ke dalam omong kosong.
- **Kanan** - imbalan yang dibentuk (imbalan mentah minus penalti KL) melacak imbalan mentah dengan erat pada awalnya, lalu tertinggal saat KL naik. Penalti sedang melakukan tugasnya: membuat model "membayar" karena melayang terlalu jauh.

Dalam sistem RLHF nyata, Anda menyetel koefisien KL sampai skor tetap naik tetapi model tetap koheren. Penalti yang terlalu kecil dan model akan meretas imbalan dengan mengeluarkan frasa pengulangan yang aneh. Terlalu besar dan model tidak pernah membaik.

## Di Mana Posisi Ini Dalam Pipeline RLHF

Ini adalah langkah kedua dari resep RLHF klasik:

1. Latih model imbalan dari preferensi.
2. **Fine-tune model bahasa dengan PPO menggunakan model imbalan tersebut.**
3. (Opsional) Lewati langkah 2 dengan DPO jika Anda ingin jalur yang lebih sederhana.

PPO adalah kuda beban yang digunakan perusahaan seperti OpenAI dan Anthropic untuk gelombang pertama model selaras, termasuk InstructGPT dan ChatGPT asli.

## Mengapa Ini Penting di Luar Lab

Pola "meningkatkan, tetapi tidak melayang" muncul di mana-mana:

- Seorang pianis yang melatih bagian yang sulit tidak mengubah seluruh tekniknya untuk menguasai satu bagian tersebut - itu akan merusak sisa resitalnya.
- Sebuah perusahaan yang mengubah situs web untuk meningkatkan pendaftaran tetap harus menjaga mereknya agar tetap dikenali oleh pengguna yang ada.
- Sebuah pabrik yang menyesuaikan satu tombol dalam suatu proses tetap menjaga tombol lainnya dekat dengan pengaturan yang sudah diketahui baik.

PPO hanyalah versi hati-hati dari ide universal ini, yang ditulis dalam matematika.

## Ringkasan Satu Kalimat

**PPO fine-tuning mendorong model menuju imbalan yang lebih tinggi sementara penalti KL menjaganya tetap dekat dengan perilaku aslinya - tingkatkan, tetapi tetaplah menjadi diri sendiri.**
