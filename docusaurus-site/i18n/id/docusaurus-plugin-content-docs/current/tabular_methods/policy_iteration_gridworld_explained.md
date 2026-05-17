# Iterasi Kebijakan untuk GridWorld 🗺️

## Apa Itu?

Bayangkan Anda sedang memainkan permainan papan pada **kisi 4×4** (seperti papan catur kecil). Anda mulai di satu sudut dan harus mencapai sudut lainnya. Setiap langkah memakan biaya 1 poin (Anda tidak ingin membuang-buang langkah!), dan mencapai tujuan tidak memberi Anda poin tambahan — Anda hanya ingin sampai di sana secepat mungkin.

**Iterasi Kebijakan (Policy Iteration)** adalah cara komputer mencari tahu **gerakan terbaik untuk setiap kotak** di papan — semuanya sekaligus!

---

## Ide Besar: Dua Langkah, Berulang-ulang

Pikirkan seperti membersihkan kamar Anda dengan seorang pembantu:

1. **Langkah 1 — Cari tahu seberapa bagus setiap kotak (Evaluasi Kebijakan)**
   Pembantu Anda berjalan di setiap kotak dan menuliskan: "Jika saya mengikuti rencana saat ini, berapa langkah yang saya perlukan untuk mencapai pintu keluar dari sini?" Mereka melakukan ini berulang kali sampai angkanya berhenti berubah.

2. **Langkah 2 — Perbaiki rencana (Peningkatan Kebijakan)**
   Sekarang Anda melihat setiap kotak dan bertanya: "Apakah ada arah yang lebih baik yang bisa saya tuju dari sini?" Jika ya, perbarui rencananya!

Ulangi Langkah 1 dan 2 sampai rencananya berhenti berubah — itulah **kebijakan optimal**!

**Contoh kehidupan nyata:** Bayangkan mencari rute tercepat ke sekolah. Pertama Anda menebak rute dan menghitung waktunya (Langkah 1). Kemudian Anda melihat setiap sudut jalan dan bertanya "apakah ada jalan pintas dari sini?" (Langkah 2). Anda memperbarui rute Anda dan ulangi sampai Anda tidak dapat menemukan jalan pintas lagi!

---

## Apa yang Ditemukan Kode Kami

GridWorld 4×4 kami memiliki dua status terminal (sudut), dan agen membayar -1 per langkah. Iterasi kebijakan konvergen hanya dalam **4 putaran** (total 139 sapuan evaluasi):

```
Nilai Status V(s):       Kebijakan Optimal:
 0.0  -1.0  -1.9  -2.7    T   ←   ←   ↓
-1.0  -1.9  -2.7  -1.9    ↑   ↑   ↑   ↓
-1.9  -2.7  -1.9  -1.0    ↑   ↑   ↓   ↓
-2.7  -1.9  -1.0   0.0    ↑   →   →   T
```

**Nilai-nilainya sangat masuk akal!** Kotak di sebelah terminal memiliki nilai -1 (satu langkah lagi). Kotak dua langkah jauhnya memiliki nilai -1.9 (= -1 + 0,9 × -1), dan seterusnya.

---

## Contoh Kehidupan Nyata

- **Navigasi GPS**: Mencari tahu belokan terbaik di *setiap* persimpangan di peta.
- **Kontrol lift**: Lantai mana yang harus dituju lift ketika ada banyak permintaan?
- **Robot pabrik**: Merencanakan jalur paling efisien di sekitar kisi gudang.

---

## Kata Kunci untuk Diingat

- **Policy**: Rencana — tindakan apa yang harus diambil di setiap status
- **Fungsi Nilai V(s)**: Seberapa baik berada di status s (lebih tinggi = lebih dekat ke tujuan)
- **Evaluasi Kebijakan (Policy Evaluation)**: Menghitung seberapa baik rencana saat ini
- **Peningkatan Kebijakan (Policy Improvement)**: Membuat rencana lebih baik menggunakan fungsi nilai
- **Kebijakan Optimal**: Rencana terbaik yang mungkin — tidak bisa diperbaiki lagi

Ide besarnya: **Anda tidak perlu mencoba setiap rencana yang mungkin! Terus perbaiki rencana saat ini, dan Anda akan menemukan rencana terbaik dalam sangat sedikit putaran.**
