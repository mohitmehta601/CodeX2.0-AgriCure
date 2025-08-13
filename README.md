# 🌱 Smart Fertilizer Recommendation System

**AGRICURE = Productivity + Profit + Planet**

An IoT-powered, AI/ML-driven system that provides **real-time, precise, and crop-specific fertilizer recommendations** to farmers. Our goal is to **boost yields, lower costs, and ensure sustainable farming** — all through data-driven, location-aware insights.

---

## 🚜 Problem Statement

Farmers often lack **accurate information** about the quality and quantity of fertilizers required for their crops.  
This leads to:

- **Nutrient imbalance** in soils
- **Reduced crop yield** and soil degradation
- **Environmental pollution** from excess fertilizer use
- Rising **production costs** (fertilizers account for ~18% for small farmers)
- Inefficient farming practices causing **economic losses**

**Stats that matter:**

- Over **40–60% of applied nitrogen and phosphorus** never reaches plants — instead, it pollutes waterways or escapes as greenhouse gases.
- **70% of Indian soils** are nutrient-deficient.
- Climate resilience and precision agriculture are now critical for food security.

---

## 💡 Solution Overview

Our **Smart Fertilizer Recommendation System** integrates **IoT sensors + Machine Learning** to deliver **real-time advisory** to farmers.

1. **IoT Sensors** measure soil NPK, moisture, pH, temperature, and humidity.
2. **ML algorithms** compare real-time readings with ideal requirements for the selected crop.
3. **System outputs** the exact fertilizer type and quantity needed — **anytime, anywhere**.

**Key Benefits:**

- Boost yield by **20–30%**
- Reduce input costs via **variable rate application (VRA)**
- Promote **climate-smart, sustainable practices**
- Scalable to millions of smallholders globally

---

## 🖥️ Tech Stack

### **Frontend**

- React.js + TypeScript – Dynamic, multilingual UI
- Tailwind CSS – Modern styling & responsiveness

### **Backend & APIs**

- Python (Flask / FastAPI) – ML model deployment & API serving
- Node.js – Supplementary API services

### **Machine Learning**

- Python + Scikit-learn – Model training & evaluation
- Precision & adaptable recommendations

### **Database & Cloud**

- Supabase – Crop, user, & sensor data storage
- ThingSpeak – Real-time IoT data ingestion

### **Hardware**

- **ESP32** – IoT controller, Wi-Fi connectivity
- **NPK Sensor (RS485)** – Soil nutrient detection (Modbus CRC)
- **Soil Moisture Sensor** – Analog water content measurement
- **DHT11** – Temperature & humidity sensor
- **SH1106 OLED** – Real-time display
- RS485–TTL converter & rechargeable power supply

---

## 🔄 System Workflow

1. **Data Collection:** IoT sensors send soil/environment data to ESP32.
2. **Cloud Sync:** Data uploaded to ThingSpeak.
3. **ML Analysis:** Backend processes and predicts fertilizer needs.
4. **User Dashboard:** Farmer gets recommendations via web-app in preferred language.
5. **Precision Delivery:** Supports Variable Rate Application techniques.

---

## 🌍 Target Users

- Small & medium-scale farmers _(1–10+ hectares)_
- Commercial farms optimizing input use
- Agribusinesses & cooperatives offering farm advisory

---

## 📊 Market & Impact

- **14 crore hectares** cultivable land in India
- **₹31K Cr** projected Indian Agri-Tech market by 2033
- Precision delivery = lower waste + higher yield
- Global potential across climate-varying regions

---

## ✨ Features

- 🌐 **Multi-language support:** English, Hindi, Punjabi & more
- ⚡ **Instant sensing:** Real-time soil & climate insights
- ☁ **Climate-smart adaptation:** Weather-aware guidance
- 📈 **Data-driven:** Advanced ML-based predictions
- 🎯 **Variable Rate Application:** Efficient nutrient usage
- 🖥 **User-centric dashboard:** Alerts, manual override, easy usability

---

## 🛤 Path to Growth

**Upcoming:**

- AI-powered crop rotation planning
- Real-time alerts for nutrient runoff & soil quality shifts

**Expansions:**

- Climate-specific configurations for new regions
- Partnership with agri-drone companies for **precision spraying**

**Long-term Vision:**

> Become the **digital backbone for precision farming** — boosting productivity, profitability & sustainability worldwide.

---

## 🚀 Getting Started

### Prerequisites
- Node.js & npm
- Python 3.x
- ESP32 board & required sensors

### Installation
