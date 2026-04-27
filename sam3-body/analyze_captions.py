import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# 設定字體與樣式
plt.style.use('ggplot')
sns.set_palette("pastel")

csv_path = '/media/ee303/4TB/sam3-body/sam3_labeled.csv'
df = pd.read_csv(csv_path)

# 定義正規表達式提取 caption 中的 gender, yaw_desc, pitch_desc
# 格式: A {gender} {yaw_desc} and {pitch_desc}
regex_pattern = r"^A (?P<gender>\S+) (?P<yaw_desc>.+?) and (?P<pitch_desc>.+)$"

# 建立新的欄位記錄解析結果
parsed_data = df['caption'].str.extract(regex_pattern)
df = pd.concat([df, parsed_data], axis=1)

# 移除無法正確解析的資料 (如果有)
df_clean = df.dropna(subset=['gender', 'yaw_desc', 'pitch_desc']).copy()

# 將 yaw_desc 中的 his 與 her 統一代換為 their，避免因為性別影響動作分類
df_clean['yaw_desc'] = df_clean['yaw_desc'].str.replace(r'\b(?:his|her)\b', 'their', regex=True)

print(f"總資料筆數: {len(df)}")
print(f"成功解析筆數: {len(df_clean)}")

# ---- 繪圖函數 ----
def plot_stats(series, title, filename_prefix):
    counts = series.value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 直方圖 (長條圖)
    sns.barplot(x=counts.values, y=counts.index, ax=axes[0])
    axes[0].set_title(f'{title} - Bar Chart')
    axes[0].set_xlabel('Count')
    axes[0].set_ylabel('Category')
    
    # 圓餅圖
    axes[1].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    axes[1].set_title(f'{title} - Pie Chart')
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_stats.png')
    plt.show()

# 繪製 Gender 統計
plot_stats(df_clean['gender'], 'Gender Distribution', 'gender')

# 繪製 Yaw Description 統計
plot_stats(df_clean['yaw_desc'], 'Yaw Description Distribution', 'yaw')

# 繪製 Pitch Description 統計
plot_stats(df_clean['pitch_desc'], 'Pitch Description Distribution', 'pitch')

# 繪製組合的統計 (Gender + Yaw + Pitch)
df_clean['combination'] = df_clean['gender'] + " | " + df_clean['yaw_desc'] + " | " + df_clean['pitch_desc']
top_combinations = df_clean['combination'].value_counts().head(10) # 顯示前 10 種組合

plt.figure(figsize=(12, 8))
sns.barplot(x=top_combinations.values, y=top_combinations.index)
plt.title('Top 10 Combinations (Gender | Yaw | Pitch)')
plt.xlabel('Count')
plt.ylabel('Combination')
plt.tight_layout()
plt.savefig('top_combinations.png')
plt.show()

print("\n--- 各類別統計數量 ---")
print("\n[Gender]")
print(df_clean['gender'].value_counts())
print("\n[Yaw Description]")
print(df_clean['yaw_desc'].value_counts())
print("\n[Pitch Description]")
print(df_clean['pitch_desc'].value_counts())
