import os
import math
import argparse
import textwrap  # 新增：用於處理文字自動換行
from PIL import Image, ImageDraw, ImageFont

def process_images(folder, cols, output_path):
    if not os.path.isdir(folder):
        print(f"錯誤: 找不到資料夾 {folder}")
        return

    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(valid_exts)]

    if not image_files:
        print("警告: 該資料夾中沒有找到支援的圖片格式！")
        return

    cell_w, cell_h_img = 300, 300
    # 修改：將文字空間加大，以容納折行後的 Prompt (可視情況再調大)
    text_space = 120 
    padding = 15
    rows = math.ceil(len(image_files) / cols)

    bg_width = cols * (cell_w + padding) + padding
    bg_height = rows * (cell_h_img + text_space + padding) + padding
    
    result_image = Image.new('RGB', (bg_width, bg_height), 'white')
    draw = ImageDraw.Draw(result_image)

    try:
        font = ImageFont.truetype("msjh.ttc", 18)
    except IOError:
        font = ImageFont.load_default()

    for index, filename in enumerate(image_files):
        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                img.thumbnail((cell_w, cell_h_img))
                col = index % cols
                row = index // cols

                x_offset = padding + col * (cell_w + padding)
                y_offset = padding + row * (cell_h_img + text_space + padding)

                paste_x = x_offset + (cell_w - img.width) // 2
                paste_y = y_offset + (cell_h_img - img.height) // 2
                result_image.paste(img, (paste_x, paste_y))

                # --- 處理長檔名：自動折行 ---
                # 設定每行大約容納 35 個字元 (可依實際顯示效果微調)
                wrapped_text = textwrap.fill(filename, width=35)

                # 設定文字的起始 X, Y 座標
                text_x = x_offset + (cell_w // 2)
                text_y_start = y_offset + cell_h_img + 10

                # 繪製多行文字
                draw.multiline_text(
                    (text_x, text_y_start), 
                    wrapped_text, 
                    fill="black", 
                    font=font, 
                    anchor="ma",      # m=水平置中對齊圖片中心, a=垂直靠上方
                    align="center"    # 讓多行文字彼此之間也置中
                )

        except Exception as e:
            print(f"無法處理圖片 {filename}: {e}")

    result_image.save(output_path)
    print(f"✅ 成功！圖片已拼接並儲存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="圖片拼接工具 (CLI 版)")
    parser.add_argument("-f", "--folder", type=str, required=True, help="要處理的圖片資料夾路徑")
    parser.add_argument("-c", "--cols", type=int, default=15, help="欄位數")
    parser.add_argument("-o", "--output", type=str, default="result.jpg", help="輸出的圖片檔名 (預設: result.jpg)")
    
    args = parser.parse_args() 
    process_images(args.folder, args.cols, args.output)