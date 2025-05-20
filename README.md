# Deteksi_Bentuk_Objek
Berikut adalah kodingan untuk Algoritma Deteksi Bentuk Objek :
```python
import cv2
import numpy as np
from google.colab import files
from PIL import Image
import io
import matplotlib.pyplot as plt
from collections import Counter

# Upload gambar
uploaded = files.upload()

for filename in uploaded.keys():
    image_stream = io.BytesIO(uploaded[filename])
    pil_image = Image.open(image_stream).convert('RGB')
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_counts = Counter()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            shape = "Tidak Dikenal"

            if len(approx) == 3:
                shape = "Segitiga"
            elif len(approx) == 4:
                ratio = w / float(h)
                if 0.95 < ratio < 1.05:
                    shape = "Persegi"
                else:
                    shape = "Kotak/Trapesium"
            elif len(approx) > 4:
                shape = "Lingkaran"

            shape_counts[shape] += 1

            # Gambar kontur dan label bentuk
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            cv2.putText(image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Tampilkan gambar dengan anotasi
    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Deteksi Bentuk Geometris")
    plt.axis("off")
    plt.show()

    # Tampilkan ringkasan jumlah bentuk
    print("📌 Ringkasan Deteksi Bentuk:")
    if shape_counts:
        for shape, count in shape_counts.items():
            print(f"- {count} {shape}")
    else:
        print("Tidak ada bentuk dikenali.")

    break  # hanya proses satu gambar


```
Hasil Output: 

![download (7)](https://github.com/user-attachments/assets/1954a0e4-e802-42dc-bc6f-f063ce56e7fd)
