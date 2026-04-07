# Yaw / Pitch Viewer

這是一個專門給 `deepfahsion.csv` 這種格式用的本機檢視 app。

支援功能:

- `yaw / pitch` 散點圖
- 點散點直接看對應圖片
- `status / person_count / 檔名` 篩選
- 滾輪、方向鍵切換圖片
- 新分頁開原圖

## 啟動

```bash
cd /media/ee303/4TB/sam3-body/sam-3d-body/yaw_pitch_viewer
python3 app.py
```

預設會讀:

```text
/media/ee303/4TB/sam3-body/sam-3d-body/deepfahsion.csv
```

開啟瀏覽器到:

```text
http://127.0.0.1:8123
```

如果之後要換別的同格式 CSV:

```bash
python3 app.py --csv /path/to/your.csv --port 9000
```
