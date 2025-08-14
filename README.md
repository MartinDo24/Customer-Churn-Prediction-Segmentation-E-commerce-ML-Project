# Customer Churn Prediction & Segmentation – E-commerce-ML-Project

Author: Đỗ Hoàng Minh

Date: 2025-04-25

Tools Used: Python (google colab)

## 📑 Table of Contents:

1.[📌Background & Overview](#-background--overview)

2.[📂 Dataset Description & Data Structure](#-dataset-description--data-structure)

3.[🔎 Final Conclusion & Recommendations](#-final-conclusion--recommendations)

## 📌 Background & Overview

### Objective:

### 📖 What is this project about? 

Customer churn is one of the most critical challenges for e-commerce companies. Losing customers not only decreases revenue but also increases marketing costs to acquire new customers. The company wants to predict potential churned users in order to offer special promotions and retain them.

This project focuses on:

- Identifying behavioral patterns of churned customers.

- Building a machine learning model to predict churn.

- Segmenting churned customers into different groups for targeted marketing.

### 👤 Who is this project for? 

- E-commerce business owners who want to reduce customer churn.

- Marketing teams aiming to design targeted retention campaigns.

- Data scientists & analysts looking to understand churn behavior and build predictive models.

- Product managers seeking insights into user engagement and satisfaction.

### GOALS
- Analyze the behavior of churned users and provide actionable recommendations.

- Build and fine-tune a Random Forest model for churn prediction.

- Segment churned users into groups and analyze differences between them.

## 📂 Dataset Description & Data Structure

- Main table: churn_prediction.xlsx

<details>
<summary> Table Description </summary>

| Variable                      | Description                               |
| ----------------------------- | ----------------------------------------- |
| `CustomerID`                  | Unique customer ID                        |
| `Churn`                       | Churn flag (1 = churned, 0 = retained)    |
| `Tenure`                      | Tenure of customer in organization        |
| `PreferredLoginDevice`        | Preferred login device                    |
| `CityTier`                    | City tier (1, 2, 3)                       |
| `WarehouseToHome`             | Distance from warehouse to home           |
| `PreferredPaymentMethod`      | Preferred payment method                  |
| `Gender`                      | Gender of customer                        |
| `HourSpendOnApp`              | Hours spent on app/website                |
| `NumberOfDeviceRegistered`    | Total registered devices                  |
| `PreferredOrderCat`           | Preferred order category (last month)     |
| `SatisfactionScore`           | Customer satisfaction score               |
| `MaritalStatus`               | Marital status                            |
| `NumberOfAddress`             | Number of addresses registered            |
| `Complain`                    | Complaints raised last month              |
| `OrderAmountHikeFromLastYear` | % increase in order amount from last year |
| `CouponUsed`                  | Number of coupons used (last month)       |
| `OrderCount`                  | Number of orders placed (last month)      |
| `DaySinceLastOrder`           | Days since last order                     |
| `CashbackAmount`              | Average cashback (last month)             |
</details>

## ⚒️ Main Process :

### 1️⃣ Data Cleaning & Preprocessing

- Checked dataset structure, data types, and basic statistics.

- Analyzed churn rate and class imbalance.

- Explored relationships between categorical variables (e.g., PreferredPaymentMethod, CityTier) and churn rate.

- Handle missing value by replace missing value with median value

- Checked and removed duplicated rows based on CustomerID to ensure data quality

<img width="1223" height="119" alt="image" src="https://github.com/user-attachments/assets/8178df14-9d9d-474d-9fc4-3a5f92c7dcf3" />

<img width="1192" height="356" alt="image" src="https://github.com/user-attachments/assets/8250ab35-e198-41c5-b998-67a607bdbae0" />

### 2️⃣ Encoding Categorical Features & Model Training & Evaluation

Applied Label Encoding for binary categorical variables (e.g., Gender, Complain)

Split data into train (70%) and validate (15%) ,test sets (15%) .

Trained Random Forest Classifier with hyperparameter tuning using GridSearchCV.

Evaluated model :

- Accuracy

- Precision

- Recall

- F1-score

- Confusion Matrix

- Balanced_accuracy (USING)

<img width="1219" height="766" alt="image" src="https://github.com/user-attachments/assets/32542da4-39a4-47c8-a03a-d2dbb0a43b3e" />

<img width="1248" height="366" alt="image" src="https://github.com/user-attachments/assets/55a71ff3-4068-4d58-8597-2135381bea18" />


### 3️⃣ Feature Importances & Customer Segmentation (on Churned Users)

Model-based (Impurity) importance: From Random Forest; quick but can be biased toward high-cardinality features.

Permutation importance (recommended): Shuffle one feature at a time and measure performance drop; reflects true predictive contribution.

Top drivers (example you can verify in your results): SatisfactionScore, Tenure, OrderCount, DaySinceLastOrder, CashbackAmount.

Business mapping: Translate top features into actions (e.g., low satisfaction → service recovery workflow).

--> Permutation importance: Model-agnostic technique to assess how much each feature affects predictions.

--> Impurity importance: Tree-based importance derived from split gains; fast but sometimes biased

Subset to churners: Segment only users with Churn = 1 for actionability.

Feature selection for clustering: Use behavioral/monetary drivers (e.g., Tenure, OrderCount, CouponUsed, CashbackAmount, HourSpendOnApp, DaySinceLastOrder, SatisfactionScore).

Scaling: Standardize numeric features before distance-based clustering.

K-Means clustering:

Determine k via Elbow (inertia vs k) and Silhouette score (cohesion/separation).

Fit model; assign cluster labels.

Cluster profiling: Compute per-cluster means/medians and churn-relevant behavior:

Cluster A — Low satisfaction, low orders: High risk; needs service recovery + strong incentives.

Cluster B — Deal hunters: High CouponUsed/CashbackAmount; respond to tailored offers.

Cluster C — Previously loyal, now inactive: High Tenure but large DaySinceLastOrder; needs win-back messaging.

Offer design (examples):

A: Proactive support ticket + expedited shipping voucher.

B: Tiered coupon/cashback bundles with minimum spend nudges.

C: Personalized win-back: “We miss you” + category-based recommendations + time-boxed credit

--? Standardization: Transform features to zero mean / unit variance so no single scale dominates distance.

--? Elbow method: Pick k where inertia reduction starts to flatten

--> Silhouette score: Measures how similar a point is to its own cluster vs others (range −1 to 1).

<img width="1241" height="507" alt="image" src="https://github.com/user-attachments/assets/a9f8e6a5-c45f-4b35-82fb-03b1fa9d391a" />

<img width="1349" height="697" alt="image" src="https://github.com/user-attachments/assets/14ba4690-296f-41b3-aa69-9f1bc56708a8" />

<img width="1370" height="590" alt="image" src="https://github.com/user-attachments/assets/df00e674-b6c4-42f8-83d3-6e55e272712d" />

Churn occurs more often at tenure < 10

Remaining users are more distributed, occurring earlier in the life cycle

<img width="1267" height="629" alt="image" src="https://github.com/user-attachments/assets/35f5b8cf-d208-4188-a57b-fa4067437b20" />

Churned users are concentrated at a fairly low CashbackAmount (<150)

Remaining users get more cashback, the key to turning into loyal users

<img width="1226" height="709" alt="image" src="https://github.com/user-attachments/assets/12968bc9-a4b3-476d-8677-8ad34fd34ed1" />

The churn rate in the group with complaints is many times higher than in the group without complaints, which shows that there is a problem with customer service.

<img width="1241" height="683" alt="image" src="https://github.com/user-attachments/assets/dfced0ee-af21-4658-b70d-01495bc03583" />

Churn users have significantly higher DaySinceLastOrder → the longer they don't come back, the higher the risk of leaving

<img width="1389" height="603" alt="image" src="https://github.com/user-attachments/assets/9f26bc17-c716-49d8-817c-a2d3d107b40b" />

People who live further away (30–50+ km) are more likely to churn.

Could be due to:

- Long delivery time

- High shipping costs

<img width="1317" height="562" alt="image" src="https://github.com/user-attachments/assets/bcc6805d-3414-427d-9189-e2a819d90a4b" />

<img width="1280" height="451" alt="image" src="https://github.com/user-attachments/assets/b402dfa8-b7d8-4fcc-bb9f-118c0078cce4" />

<img width="1268" height="759" alt="image" src="https://github.com/user-attachments/assets/7021a05a-9409-4a2b-a541-408be9baa1c1" />

Filter out churn users for evaluation

<img width="1237" height="380" alt="image" src="https://github.com/user-attachments/assets/523d117c-2cd2-433b-97f2-57fb25c98e82" />

Show Feature Importances

<img width="1236" height="496" alt="image" src="https://github.com/user-attachments/assets/660bbb2b-77ab-4344-b1d4-df2e465e2abc" />

CashbackAmount plays a major role in causing users to leave, accounting for more than 20%

<img width="1242" height="666" alt="image" src="https://github.com/user-attachments/assets/c1e69464-8f2c-46d2-ba8e-40b3abdb104a" />

<img width="1235" height="685" alt="image" src="https://github.com/user-attachments/assets/45fdc87f-cb0e-4790-87d4-161eca071ee4" />

Cluster 0 – High-Value Churners:
CashbackAmount: Highest (190-210)
OrderCount: Highest (7-10 orders)
--> This is a group of customers who have bought a lot and received high refunds, meaning they were very active, but for some reason left.

Cluster 1 – Low-Engaged Churners:
CashbackAmount: Lowest (130-140)
OrderCount: Lowest (~1.5 orders)
--> This is a group of customers who are almost inactive, have few orders, and rarely receive cashback. They may never feel that the product/service is attractive enough.

Cluster 2 – Medium-Value Churners:
CashbackAmount: Medium (160-175)
OrderCount: Medium (~2 orders)
--> This group has average engagement and spending. They may return if given the right push.

## 🔎 Final Conclusion & Recommendations

Final Conclusion

- The analysis identified clear behavioral differences between churned and retained customers.

- Key churn drivers include CashbackAmount, SatisfactionScore, DaySinceLastOrder, OrderCount, and short Tenure.

- The Random Forest model, after fine-tuning, achieved strong recall, ensuring the business can capture the majority of at-risk customers for retention campaigns.

- Segmentation of churned users revealed three distinct customer profiles, each requiring different engagement strategies.

- These insights bridge the gap between predictive analytics and targeted marketing actions.

Recommendations

1. Boost Satisfaction for At-Risk Customers

- Deploy proactive customer support for users with low satisfaction.

- Offer instant resolution channels (live chat, VIP hotline).

2. Re-Engage Inactive Loyal Customers

- Send personalized win-back campaigns highlighting previously purchased categories.

- Provide time-limited cashback credits to encourage immediate purchase.

3. Target Deal Hunters Efficiently

- Offer tiered discounts/cashback with a minimum spend requirement.

- Use A/B testing to find optimal offer value that maximizes retention without eroding margins.

4. Reduce Early-Stage Churn

- Implement onboarding campaigns with app tutorials, first-purchase coupons, and push notifications.

- Incentivize second purchase within the first 30 days.

5. Continuous Model Monitoring

- Retrain churn model quarterly to adapt to evolving customer behaviors.

- Track KPI impact: retention rate uplift, revenue saved from prevented churn.

