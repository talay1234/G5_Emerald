# Emerald: เว็บ AI เพื่อการจำแนกงูพิษในไทย

![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-blue)
![Python](https://img.shields.io/badge/Backend-Flask-lightgrey)
![TensorFlow](https://img.shields.io/badge/Model-EfficientNetV2-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

**Emerald** คือเว็บสำหรับจำแนกงูพิษ 7 ชนิดที่พบบ่อยในประเทศไทย โดยใช้เทคโนโลยีปัญญาประดิษฐ์ (AI) และการเรียนรู้เชิงลึก (Deep Learning) พัฒนาขึ้นเพื่อช่วยสนับสนุนบุคลากรทางการแพทย์และเจ้าหน้าที่กู้ภัยในการระบุชนิดของงูจากภาพถ่ายได้อย่างรวดเร็วและแม่นยำ เพื่อลดความเสี่ยงและเพิ่มประสิทธิภาพในการรักษาผู้ป่วยที่ถูกงูกัด

---

## ที่มาและความสำคัญ (Motivation & Pain Points)
ปัญหาการจำแนกงูพิษในประเทศไทยคือ **งูพิษ 7 สายพันธุ์ที่พบบ่อย** มีลักษณะบางอย่างที่คล้ายคลึงกัน หรือคล้ายกับงูไม่มีพิษ ทำให้การระบุชนิดด้วยสายตาเปล่าอาจเกิดความผิดพลาด ซึ่งส่งผลกระทบโดยตรงต่อ:
* **การรักษา:** การเลือกใช้เซรุ่ม (Antivenom) ผิดชนิด หรือการรักษาที่ล่าช้า
* **ความปลอดภัย:** ความเสี่ยงต่อชีวิตของผู้ป่วยและเจ้าหน้าที่

กลุ่มผู้จัดทำจึงพัฒนาโครงการนี้ขึ้นเพื่อเป็นเครื่องมือช่วยจำแนกชนิดงู (Classification) เพื่อให้แพทย์และเจ้าหน้าที่สามารถตัดสินใจวางแผนการรักษาได้อย่างถูกต้องและทันท่วงที

---

## ขอบเขตการทำงาน (Scope)
ระบบรองรับการจำแนกงูพิษที่มีนัยสำคัญทางการแพทย์ในไทย 7 ชนิด และงูไม่มีพิษ (สำหรับเปรียบเทียบ) รวม 8 Class ดังนี้:

**กลุ่มงูพิษ (Venomous):**
1.  งูสามเหลี่ยม (Banded Krait)
2.  งูแมวเซา (Eastern Russell's Viper)
3.  งูเขียวหางไหม้ (Green Pit Viper)
4.  งูจงอาง (King Cobra)
5.  งูทับสมิงคลา (Malayan Krait)
6.  งูกะปะ (Malayan Pit Viper)
7.  งูเห่า (Monocled Cobra)

**กลุ่มงูไม่มีพิษ (Non-Venomous):**
* งูสิง (Indo-Chinese Rat Snake) - *ใช้เป็นตัวแทนงูไม่มีพิษที่มีลักษณะคล้ายงูเห่า*

---

## เทคโนโลยีที่ใช้ (Tech Stack)

### AI & Backend
* **Model:** EfficientNetV2 (CNN Architecture)
* **Framework:** TensorFlow / Keras
* **Backend:** Python (Flask API)
* **Deployment:** Vercel

### Frontend (Web Application)
* **Framework:** React (Vite) + TypeScript
* **Styling:** Tailwind CSS (Theme: Clinic Clean + Terra Neutral)
* **Features:** ถ่ายภาพ/อัปโหลดภาพ, แสดงผล Real-time, บันทึกประวัติ (History)

---

## คณะผู้จัดทำ (Team Members)
**Sec 24 - กลุ่มที่ 5**

| รหัสนักศึกษา | ชื่อ-นามสกุล | หน้าที่ความรับผิดชอบ (Roles) |
|---|---|---|
| 66010262 | นางสาวณัฐรัตน์ เรืองปิยะเสรี | Dataset, AI Model, Report, Presentation Slides |
| 66010474 | นางสาวปวิชญา อ่อนอำไพ | Dataset, Web (Frontend), Clip, Presentation Slides, Pitching |
| 66010727 | นางสาววนัสชาพร พลพัฒน์ | Dataset, AI Model, Presentation Slides |
| 66011448 | นายภัทรดนัย จำรัส | AI Model, Backend & Deploy, Clip, Report |
| 66011456 | นางสาวมนพร พรหมมงคลกุล | Dataset, AI Model, Presentation Slides |

---

## แนวทางการพัฒนาในอนาคต (Future Work)
1.  **Model Optimization:** ปรับปรุงความแม่นยำด้วยการเพิ่มชั้น Layer หรือทำ Hyperparameter Tuning ให้โมเดลเรียนรู้ละเอียดขึ้น
2.  **Extended Classification:** เพิ่มชนิดของงูให้ครอบคลุมทั้งงูพิษและไม่มีพิษสายพันธุ์อื่นๆ ในไทย เพื่อลดความลำเอียงของข้อมูล
3.  **Mobile Application:** พัฒนาเป็นแอปพลิเคชันบนสมาร์ตโฟนเพื่อให้ใช้งานสะดวกในพื้นที่เกิดเหตุ
4.  **System Integration:** เชื่อมต่อระบบข้อมูลไปยังศูนย์กู้ภัยหรือโรงพยาบาลโดยตรง

---
