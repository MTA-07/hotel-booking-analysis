import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense



os.makedirs("outputs", exist_ok = True)
os.makedirs("charts", exist_ok = True)


df = pd.read_csv("data/hotel_bookings.csv")
print("--- ILK 5 SATIR ---")
print(df.head())

print("\n--- VERI BILGISI ---")
print(df.info())

print("\n --- TABLO BOYUTU ---")
print(df.shape)

print("\n --- EKSIK VERI ANALIZI ---")

missing_report = pd.DataFrame({
    "Eksik_Sayisi": df.isnull().sum(),
    "Eksik_Yuzdesi": (df.isnull().sum() / len(df)) * 100
})
missing_report = missing_report[missing_report["Eksik_Sayisi"]>0]
missing_report.to_csv("outputs/missing_report.csv")
print(missing_report)

print("\n --- VERI TEMIZLEME ---")
df["children"] = df["children"].fillna(0)
df["country"] = df["country"].fillna("Unknown")

if "agent" in df.columns and "company" in df.columns:
    df = df.drop(["agent", "company"], axis = 1)

print("Eksik veriler dolduruldu ve gereksiz kolonlar silindi")

print("\n YENI OZELLIKLER URETME ---")

df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
df["total_guests"] = df["adults"] + df["children"] + df["babies"]
df["is_family"] = np.where(df["total_guests"] > 2, 1, 0)
df["stay_type"] = np.where(df["total_nights"] > 5, "long", "short")

print("Yeni ozellikler basariyla olusturuldu")

print("\n --- STRATEJIK RAPORLAR HAZIRLANIYOR ---")
hotel_summary = df.groupby("hotel").agg({
    "is_canceled": "mean",
    "total_nights": "mean",
    "adr": "mean"
})
hotel_summary["is_canceled"] = hotel_summary["is_canceled"] * 100
hotel_summary.to_csv("outputs/hotel_summary.csv")
customer_report =  df.groupby("customer_type").agg({
    "is_canceled": "mean",
    "total_nights": "mean",
    "adr": "mean"
})
customer_report["is_canceled"] = customer_report["is_canceled"] * 100
customer_report.to_csv("outputs/customer_report.csv")
print("Raporlar basariyla 'outputs' klasorune kaydedildi")

print("\n --- GRAFIKLER CIZILIYOR VE KAYDEDILIYOR ---")

monthly_bookings = df["arrival_date_month"].value_counts()

plt.figure(figsize = (10,5))
monthly_bookings.plot(kind = "bar", color = "skyblue")
plt.title("Aylara Gore Toplam Rezervasyon Sayilari")
plt.xlabel("Aylar")
plt.ylabel("Rezervasyon Sayisi")
plt.xticks(rotation = 45)
plt.tight_layout()
plt.savefig("charts/monthly_bookings.png")
plt.close()

hotel_cancel_rate = df.groupby("hotel")["is_canceled"].mean() * 100

plt.figure(figsize = (7, 5))
hotel_cancel_rate.plot(kind = "bar", color = "salmon")
plt.title("Otel Iptal Oranlari")
plt.xlabel("Otel Tipi")
plt.ylabel("Iptal Orani (%)")
plt.xticks(rotation = 0)
plt.tight_layout()
plt.savefig("charts/hotel_cancel_rate.png")
plt.close()

top_countries = df["country"].value_counts().head(10)
plt.figure(figsize = (10, 5))
top_countries.plot(kind = "bar", color = "lightgreen")
plt.title("En Cok Rezervasyon Yapan 10 Ulke")
plt.xlabel("Ulke Kodu")
plt.ylabel("Rezervasyon Sayisi")
plt.xticks(rotation = 0)
plt.tight_layout()
plt.savefig("charts/country_top10.png")
plt.close()


print("Tum grafikler cizildi ve 'charts' klasorune kaydedildi")


#BEYIN CERRAHISI KISMI 


features = [
    "lead_time",
    "arrival_date_year",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "total_nights",
    "total_guests",
    "is_family",

]

x = df[features] # bu musterinin kac gun onceden rezervasyon yaptigini kac gece kalacagini daha once iptal edip etmedigini biliyoruz (anlami)
y = df["is_canceled"] # gelecekte iptal edip etmeyecegini de kestiriyorum falan

x = x.replace([np.inf, -np.inf], np.nan)
x = x.fillna(0)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = Sequential([
    Dense(64, activation="relu", input_shape=(x_train_scaled.shape[1],)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train_scaled,
    y_train,
    epochs=20,
    batch_size=32, #32 serli gruplar halinde ogrenecek 119 bin satirin hepsini ayni anda ogrenmek yerine
    validation_split=0.2
)

test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test)

print('Test Loss:', test_loss)
print("Test Accuracy:", test_accuracy)

print("\n--- EGITIM GRAFIGI HAZIRLANIYOR ---")


plt.figure(figsize=(12,5))

#Accuracy Grafigi
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Egitim Basarisi")
plt.plot(history.history["val_accuracy"], label = "Dogrulama Basarisi")
plt.title("Model Dogrulugu")
plt.xlabel("Tur (Epoch)")
plt.ylabel("Dogruluk")
plt.legend()

#Loss Grafigi
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Egitim Hatasi")
plt.plot(history.history["val_loss"], label= "Dogrulama Hatasi")
plt.title("Model Hatasi (Loss)")
plt.xlabel("Tur (Epoch)")
plt.ylabel("Hata")
plt.legend()

plt.tight_layout()
plt.savefig("charts/training_history.png")
plt.show()

print("Egitim grafigi 'charts/training_history.png' olarak kaydedildi")



print("\n--- ORNEK TAHMIN UYGULAMASI ---")
# simdi test asamasindayiz
#hayali bir musteri olusturalim (sayisal kolon sirasiyla)
# features listesindeki sirayla ayni yapmaliyiz

sample_customer = np.array([[
    150,  #lead_time: kac gun onceden rezervasyon yaptigi
    2026, #arrival_date_year
    2,    #stays_in_weekend_nights
    5,    #stays_in_week_nights
    2,    #adults
    0,    #children
    0,    #babies
    0,    #is_repeated_guest
    1,    #previous_cancellations
    0,    #previous_bookings_not_canceled
    0,    #booking_changes
    0,    #days_in_waiting_list
    120.0,#adr: gunluk odedigi fiyat
    0,    #required_car_parking_spaces
    1,    #total_of_special_request
    7,    #total_nights
    2,    #total_guests
    0     #is_family
]])

# simdi bu veriyi scaler kullanarak olceklendirmeliyiz
sample_scaled = scaler.transform(sample_customer)
#tahmin
prediction = model.predict(sample_scaled)
probability = prediction[0][0] * 100

print(f"\n Bu musterinin rezervasyonunu iptal etme olasiligi: %{probability:.2f}")

if probability > 50:
    print("SONUC: BU REZERVASYON RISKLI! (Iptal Bekleniyor)")
else:
    print("SONUC: BU REZERVASYON GUVENLI! (Konaklama Bekleniyor)")




