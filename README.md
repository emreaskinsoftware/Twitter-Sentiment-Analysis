# NLP Projesi: İngilizce Oyun Tweet'leri Duygu Analizi

**Amaç:** Bu proje, Kaggle'dan alınan bir tweet veri setini kullanarak, bir tweet'in içeriğine (metnine) bakarak "Pozitif" veya "Negatif" olduğunu **%92'nin üzerinde bir doğrulukla** tahmin eden bir Doğal Dil İşleme (NLP) modeli geliştirmeyi amaçlamaktadır.

**Portföydeki Etkisi:** Bu proje, baştan sona profesyonel bir NLP iş akışını uygulama becerimi göstermektedir:
* **Veri Keşfi (EDA):** Veri setinin varsayılan etiketlerini sorgulama ve gerçek içeriğini (İngilizce oyun tweet'leri) keşfetme.
* **Metin Temizleme (Preprocessing):** `Regex` (Regular Expressions) kullanarak link, @mention, noktalama ve sayılar gibi gürültüleri temizleme.
* **Veri Filtreleme:** Analiz amacına uymayan ("Irrelevant", "Neutral") sınıfları bilinçli olarak filtreleyerek modelin performansını artırma.
* **Metin Vektörleştirme (TF-IDF):** `Scikit-learn`'ün `TfidfVectorizer` aracını kullanarak metin verisini makine öğrenimine uygun sayısal vektörlere dönüştürme.
* **Profesyonel Model Değerlendirme:** Modeli `train_test_split` yerine, **tamamen ayrı bir validasyon (test) seti** (`twitter_validation.csv`) üzerinde test ederek modelin "genelleme" (generalization) yeteneğini dürüstçe ölçme.

**Kullanılan Araçlar:**
* Python
* Pandas (Veri yükleme, filtreleme ve manipülasyon)
* Scikit-learn (`TfidfVectorizer`, `LogisticRegression`, `accuracy_score`, `classification_report`)
* Regex (`re`) ve `string` (Metin temizleme için)
* Matplotlib & Seaborn (İlk EDA için)

---

## 🧭 Analiz ve Modelleme İş Akışı

### 1. Keşif, Temizlik ve Filtreleme

1.  **Veri Keşfi:** Proje, Kaggle'da "Türkçe NLP" olarak etiketlenmiş bir veri seti ile başladı. Ancak, `df.head()` ile yapılan ilk Keşifçi Veri Analizi (EDA) sonucunda, metinlerin (`im getting on borderlands...`) **İngilizce** olduğu ve `topic` sütununa (Borderlands, CallOfDuty, vb.) bakıldığında **video oyunları** ile ilgili olduğu **tespit edildi.** Projenin yönü bu keşfe göre "İngilizce Oyun Tweet'leri Analizi" olarak güncellendi.
2.  **Veri Temizleme:** Ham metin verisi (`text` sütunu), modelin kafasını karıştıracak gürültüler içeriyordu. `Regex` kullanılarak bu metinler için bir `clean_text` fonksiyonu yazıldı:
    * Tüm metin küçük harfe çevrildi.
    * Linkler (http...), @mention'lar, sayılar ve noktalama işaretleri kaldırıldı.
3.  **Veri Filtreleme (Kritik Adım):** Orijinal veri setinde 4 duygu sınıfı vardı (Positive, Negative, Neutral, Irrelevant).
    * **Proje Amacı:** Modelin "duygu" (sentiment) tahmin etmesi istendi.
    * **Karar:** `Irrelevant` (İlgisiz) ve `Neutral` (Nötr) sınıflarının, "duygu" belirtmeyen gürültü sınıfları olduğuna karar verildi.
    * **Eylem:** Modelin sadece "Pozitif" ve "Negatif" arasındaki net farkı öğrenmesi için bu iki sınıf veri setinden **filtrelendi**. Bu, modelin başarısını doğrudan etkileyen en önemli karar oldu.

### 2. Profesyonel Test Stratejisi

Veri seti, `twitter_training.csv` ve `twitter_validation.csv` olarak iki ayrı dosya halinde gelmişti.

Modelin gerçek dünya performansını ölçmek için, veriyi `train_test_split` ile yapay olarak bölmek yerine **daha profesyonel bir yol** izlendi:

* **Eğitim (Train):** Model, `twitter_training.csv` dosyasından hazırlanan `X_train` (43,013 tweet) üzerinde eğitildi.
* **Test (Validation):** Model, daha önce hiç görmediği ve **tamamen ayrı bir dosya** olan `twitter_validation.csv`'den hazırlanan `X_test` (543 tweet) üzerinde test edildi.

### 3. Metin Vektörleştirme (TF-IDF)

Makine öğrenimi modelleri metinle çalışamaz, sayılarla çalışır. Temizlenmiş `cleaned_text` sütununu sayısallaştırmak için `TfidfVectorizer` kullanıldı:

* `max_features=5000` parametresi ile en önemli 5000 kelime/terim (token) seçildi.
* `vectorizer`, `X_train` üzerinde **eğitildi (`.fit_transform()`)** ve bu öğrenilen sözlük, `X_test`'e **uygulandı (`.transform()`)**.

### 4. Modelleme ve Değerlendirme

Bu sınıflandırma problemi için en güvenilir temel modellerden biri olan `LogisticRegression` seçildi.

* Model, `X_train_v` (43,013 tweet'in 5000 özellikli vektörü) üzerinde eğitildi.
* Eğitilen model, `X_test_v` (543 test tweet'i) üzerinde tahmin yaptı.

---

## 📊 Sonuçlar: %92 Doğruluk

Model, daha önce hiç görmediği ve ayrı bir dosyadan gelen test verisi üzerinde **%92.08** gibi çok yüksek bir doğruluk oranına ulaştı.

Detaylı sınıflandırma raporu, modelin başarısının "şans" olmadığını ve her iki sınıfta da mükemmel bir dengeye sahip olduğunu kanıtlamaktadır:

```
--- Model Doğruluk (Accuracy) Skoru ---
92.08%
(Model, 543 test tweet'inin 500 tanesini doğru tahmin etti.)

--- Detaylı Sınıflandırma Raporu ---
              precision    recall  f1-score   support

    Negative       0.91      0.93      0.92       266
    Positive       0.93      0.91      0.92       277

    accuracy                           0.92       543
   macro avg       0.92      0.92      0.92       543
weighted avg       0.92      0.92      0.92       543
```

**Değerlendirme:** Modelin hem `Negative` (%92 f1-score) hem de `Positive` (%92 f1-score) sınıflarını eşit derecede iyi yakalaması; veri temizleme, gürültü filtreleme (`Irrelevant`/`Neutral`) ve profesyonel test metodolojisinin (`validation.csv` kullanılması) başarısını doğrulamaktadır.
