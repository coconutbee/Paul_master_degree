#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


REQUIRED_COLUMNS = {
    "image_path",
    "sam3d_head_body_yaw",
    "sam3d_head_pitch",
    "sam3d_status",
    "sam3d_person_count",
}


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"CSV is missing required columns: {missing_text}")

        rows = []
        for index, raw in enumerate(reader):
            image_path = raw["image_path"].strip()
            yaw = float(raw["sam3d_head_body_yaw"])
            pitch = float(raw["sam3d_head_pitch"])
            status = raw["sam3d_status"].strip()
            person_count_raw = raw["sam3d_person_count"].strip()
            person_count = int(float(person_count_raw))
            rows.append(
                {
                    "id": index,
                    "image_path": image_path,
                    "image_name": os.path.basename(image_path),
                    "yaw": yaw,
                    "pitch": pitch,
                    "status": status,
                    "person_count": person_count,
                    "image_exists": Path(image_path).is_file(),
                }
            )
    return rows


def build_payload(rows: list[dict]) -> dict:
    yaw_values = [row["yaw"] for row in rows] or [0.0]
    pitch_values = [row["pitch"] for row in rows] or [0.0]
    status_values = sorted({row["status"] for row in rows})
    person_counts = sorted({row["person_count"] for row in rows})
    return {
        "rows": rows,
        "meta": {
            "count": len(rows),
            "yaw_min": min(yaw_values),
            "yaw_max": max(yaw_values),
            "pitch_min": min(pitch_values),
            "pitch_max": max(pitch_values),
            "statuses": status_values,
            "person_counts": person_counts,
        },
    }


HTML = """<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Yaw / Pitch 檢視器</title>
  <style>
    :root {
      --bg: #f3efe6;
      --panel: rgba(255, 252, 245, 0.92);
      --border: #d8ccb6;
      --ink: #1d2428;
      --muted: #5d666d;
      --accent: #b3522f;
      --accent-soft: rgba(179, 82, 47, 0.14);
      --ok: #2b7a78;
      --warn: #b58900;
      --shadow: 0 18px 48px rgba(46, 35, 20, 0.12);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Noto Sans TC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(179, 82, 47, 0.16), transparent 28%),
        radial-gradient(circle at bottom right, rgba(43, 122, 120, 0.14), transparent 24%),
        linear-gradient(180deg, #f7f1e7 0%, #efe7d9 100%);
      min-height: 100vh;
    }

    .shell {
      display: grid;
      grid-template-columns: minmax(360px, 1.1fr) minmax(360px, 1fr);
      gap: 18px;
      padding: 18px;
      min-height: 100vh;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }

    .left {
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      gap: 14px;
      padding: 18px;
    }

    .right {
      display: grid;
      grid-template-rows: auto minmax(320px, 1fr) auto auto;
      gap: 14px;
      padding: 18px;
    }

    h1 {
      margin: 0;
      font-size: 28px;
      letter-spacing: 0.02em;
    }

    .subtle, label, .hint, .meta-line {
      color: var(--muted);
      font-size: 14px;
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }

    .stat {
      background: rgba(255, 255, 255, 0.7);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
    }

    .stat b {
      display: block;
      font-size: 22px;
      margin-bottom: 4px;
      color: var(--accent);
    }

    .controls {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      align-items: end;
    }

    .field {
      display: grid;
      gap: 6px;
    }

    select, input[type="text"], button {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      font: inherit;
      color: var(--ink);
      background: #fffdf8;
    }

    button {
      background: var(--accent);
      color: #fff7f2;
      border-color: transparent;
      cursor: pointer;
    }

    button.secondary {
      background: #fffaf0;
      color: var(--ink);
      border-color: var(--border);
    }

    .canvas-wrap {
      position: relative;
      padding: 12px;
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid var(--border);
      border-radius: 16px;
      min-height: 520px;
    }

    canvas {
      width: 100%;
      height: 520px;
      display: block;
      border-radius: 12px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.95), rgba(244, 239, 229, 0.95));
    }

    .image-frame {
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
      background:
        linear-gradient(135deg, rgba(255, 250, 240, 0.98), rgba(235, 242, 240, 0.98));
      display: grid;
      place-items: center;
      min-height: 320px;
    }

    .image-frame img {
      max-width: 100%;
      max-height: 70vh;
      display: block;
      object-fit: contain;
      background: #fff;
    }

    .image-empty {
      padding: 36px;
      text-align: center;
      color: var(--muted);
    }

    .detail-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .detail-card {
      background: rgba(255, 255, 255, 0.74);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
    }

    .detail-card strong {
      display: block;
      margin-bottom: 6px;
    }

    .path-box {
      border: 1px dashed var(--border);
      background: rgba(255,255,255,0.65);
      border-radius: 12px;
      padding: 12px;
      word-break: break-all;
      font-size: 13px;
      line-height: 1.4;
    }

    .toolbar {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 10px;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 8px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 13px;
    }

    .hidden { display: none; }

    @media (max-width: 1100px) {
      .shell { grid-template-columns: 1fr; }
      .controls { grid-template-columns: 1fr 1fr; }
      .toolbar { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="panel left">
      <div>
        <h1>Yaw / Pitch 品質檢視器</h1>
        <div class="subtle">點散點看圖片，確認姿態數值和實際頭部方向是否一致。支援滑鼠點選、鍵盤切換與條件篩選。</div>
      </div>

      <div class="stats">
        <div class="stat"><b id="stat-total">0</b><span>總筆數</span></div>
        <div class="stat"><b id="stat-visible">0</b><span>目前顯示</span></div>
        <div class="stat"><b id="stat-selected">-</b><span>目前索引</span></div>
      </div>

      <div class="controls">
        <div class="field">
          <label for="statusFilter">Status</label>
          <select id="statusFilter"></select>
        </div>
        <div class="field">
          <label for="personFilter">Person Count</label>
          <select id="personFilter"></select>
        </div>
        <div class="field">
          <label for="searchInput">檔名搜尋</label>
          <input id="searchInput" type="text" placeholder="例如 front 或 side">
        </div>
        <div class="field">
          <label>&nbsp;</label>
          <button id="resetBtn" class="secondary" type="button">重設篩選</button>
        </div>
      </div>

      <div class="canvas-wrap">
        <canvas id="plot" width="900" height="520"></canvas>
      </div>

      <div class="hint">
        操作: 點散點選圖，滑鼠滾輪或 <kbd>[</kbd>/<kbd>]</kbd> 依目前篩選結果切換，<kbd>f</kbd> 回到第一筆，<kbd>l</kbd> 到最後一筆。
      </div>
    </section>

    <section class="panel right">
      <div class="toolbar">
        <button id="prevBtn" class="secondary" type="button">上一筆</button>
        <button id="nextBtn" class="secondary" type="button">下一筆</button>
        <button id="openBtn" type="button">新分頁開圖</button>
      </div>

      <div class="image-frame" id="imageFrame">
        <div class="image-empty">尚未選取資料點</div>
      </div>

      <div class="detail-grid">
        <div class="detail-card">
          <strong>姿態</strong>
          <div class="meta-line">Yaw: <span id="detailYaw">-</span></div>
          <div class="meta-line">Pitch: <span id="detailPitch">-</span></div>
          <div class="meta-line">Status: <span id="detailStatus">-</span></div>
          <div class="meta-line">Person Count: <span id="detailPerson">-</span></div>
          <div class="meta-line">檔案存在: <span id="detailExists">-</span></div>
        </div>
        <div class="detail-card">
          <strong>檢視重點</strong>
          <div class="pill">Yaw 正值通常往觀者左側或右側，依你的定義檢查一致性</div>
          <div class="pill">Pitch 正負應對應抬頭 / 低頭</div>
          <div class="pill">留意 side/front 檔名是否和角度量級合理</div>
        </div>
      </div>

      <div class="path-box" id="detailPath">尚未選取圖片</div>
    </section>
  </div>

  <script>
    const state = {
      rows: [],
      filtered: [],
      selectedId: null,
      hoveredId: null,
      meta: null,
      plotPoints: [],
      dpr: window.devicePixelRatio || 1,
    };

    const canvas = document.getElementById("plot");
    const ctx = canvas.getContext("2d");
    const statusFilter = document.getElementById("statusFilter");
    const personFilter = document.getElementById("personFilter");
    const searchInput = document.getElementById("searchInput");
    const resetBtn = document.getElementById("resetBtn");
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    const openBtn = document.getElementById("openBtn");

    function fmt(value) {
      return Number(value).toFixed(2);
    }

    function scaleCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      state.dpr = dpr;
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    async function boot() {
      const response = await fetch("/api/data");
      const payload = await response.json();
      state.rows = payload.rows;
      state.meta = payload.meta;
      initFilters(payload.meta);
      applyFilters();
      bindEvents();
    }

    function initFilters(meta) {
      statusFilter.innerHTML = `<option value="">全部</option>` +
        meta.statuses.map((value) => `<option value="${value}">${value}</option>`).join("");
      personFilter.innerHTML = `<option value="">全部</option>` +
        meta.person_counts.map((value) => `<option value="${value}">${value}</option>`).join("");
      document.getElementById("stat-total").textContent = meta.count;
    }

    function applyFilters() {
      const status = statusFilter.value;
      const person = personFilter.value;
      const query = searchInput.value.trim().toLowerCase();

      state.filtered = state.rows.filter((row) => {
        if (status && row.status !== status) return false;
        if (person && String(row.person_count) !== person) return false;
        if (query && !row.image_name.toLowerCase().includes(query) && !row.image_path.toLowerCase().includes(query)) return false;
        return true;
      });

      document.getElementById("stat-visible").textContent = state.filtered.length;

      if (!state.filtered.length) {
        state.selectedId = null;
        renderDetails(null);
      } else if (!state.filtered.some((row) => row.id === state.selectedId)) {
        state.selectedId = state.filtered[0].id;
        renderDetails(state.filtered[0]);
      } else {
        renderDetails(getSelectedRow());
      }

      drawPlot();
    }

    function getSelectedRow() {
      return state.filtered.find((row) => row.id === state.selectedId) || null;
    }

    function drawPlot() {
      scaleCanvas();
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const pad = { top: 28, right: 18, bottom: 42, left: 54 };
      const innerW = width - pad.left - pad.right;
      const innerH = height - pad.top - pad.bottom;
      const yawMin = state.meta.yaw_min;
      const yawMax = state.meta.yaw_max;
      const pitchMin = state.meta.pitch_min;
      const pitchMax = state.meta.pitch_max;

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#fffdf7";
      ctx.fillRect(0, 0, width, height);

      const zeroX = projectX(0);
      const zeroY = projectY(0);

      drawGrid(width, height, pad, innerW, innerH, yawMin, yawMax, pitchMin, pitchMax);

      ctx.strokeStyle = "rgba(29, 36, 40, 0.4)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(zeroX, pad.top);
      ctx.lineTo(zeroX, height - pad.bottom);
      ctx.moveTo(pad.left, zeroY);
      ctx.lineTo(width - pad.right, zeroY);
      ctx.stroke();

      state.plotPoints = state.filtered.map((row, idx) => {
        const x = projectX(row.yaw);
        const y = projectY(row.pitch);
        const selected = row.id === state.selectedId;
        const hovered = row.id === state.hoveredId;
        const radius = selected ? 7 : hovered ? 6 : 4.5;

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = selected ? "#b3522f" : row.image_exists ? "#2b7a78" : "#b58900";
        ctx.globalAlpha = selected ? 1 : 0.82;
        ctx.fill();
        ctx.globalAlpha = 1;

        if (selected || hovered) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.stroke();
        }

        return { x, y, radius: Math.max(radius, 8), id: row.id, idx };
      });

      if (state.hoveredId != null) {
        const row = state.filtered.find((item) => item.id === state.hoveredId);
        if (row) {
          drawTooltip(row);
        }
      }

      function projectX(value) {
        if (yawMax === yawMin) return pad.left + innerW / 2;
        return pad.left + ((value - yawMin) / (yawMax - yawMin)) * innerW;
      }

      function projectY(value) {
        if (pitchMax === pitchMin) return pad.top + innerH / 2;
        return pad.top + innerH - ((value - pitchMin) / (pitchMax - pitchMin)) * innerH;
      }

      function drawTooltip(row) {
        const point = state.plotPoints.find((item) => item.id === row.id);
        if (!point) return;
        const text = `${row.image_name} | yaw ${fmt(row.yaw)} | pitch ${fmt(row.pitch)}`;
        ctx.font = "13px Segoe UI";
        const textWidth = ctx.measureText(text).width;
        const boxW = textWidth + 18;
        const boxH = 28;
        let x = point.x + 12;
        let y = point.y - 34;
        if (x + boxW > width - pad.right) x = point.x - boxW - 12;
        if (y < pad.top) y = point.y + 12;

        ctx.fillStyle = "rgba(29, 36, 40, 0.88)";
        ctx.fillRect(x, y, boxW, boxH);
        ctx.fillStyle = "#fff";
        ctx.fillText(text, x + 9, y + 18);
      }
    }

    function drawGrid(width, height, pad, innerW, innerH, yawMin, yawMax, pitchMin, pitchMax) {
      const tickCount = 6;
      ctx.strokeStyle = "rgba(93, 102, 109, 0.15)";
      ctx.fillStyle = "#5d666d";
      ctx.lineWidth = 1;
      ctx.font = "12px Segoe UI";

      for (let i = 0; i <= tickCount; i++) {
        const x = pad.left + (innerW / tickCount) * i;
        const y = pad.top + (innerH / tickCount) * i;
        const yawValue = yawMin + ((yawMax - yawMin) / tickCount) * i;
        const pitchValue = pitchMax - ((pitchMax - pitchMin) / tickCount) * i;

        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, height - pad.bottom);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(width - pad.right, y);
        ctx.stroke();

        ctx.fillText(fmt(yawValue), x - 14, height - 18);
        ctx.fillText(fmt(pitchValue), 6, y + 4);
      }

      ctx.fillStyle = "#1d2428";
      ctx.font = "bold 13px Segoe UI";
      ctx.fillText("Yaw", width / 2 - 12, height - 2);
      ctx.save();
      ctx.translate(16, height / 2 + 18);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Pitch", 0, 0);
      ctx.restore();
    }

    function renderDetails(row) {
      const frame = document.getElementById("imageFrame");
      document.getElementById("stat-selected").textContent = row ? (state.filtered.findIndex((item) => item.id === row.id) + 1) : "-";

      if (!row) {
        frame.innerHTML = `<div class="image-empty">目前篩選沒有資料</div>`;
        document.getElementById("detailYaw").textContent = "-";
        document.getElementById("detailPitch").textContent = "-";
        document.getElementById("detailStatus").textContent = "-";
        document.getElementById("detailPerson").textContent = "-";
        document.getElementById("detailExists").textContent = "-";
        document.getElementById("detailPath").textContent = "尚未選取圖片";
        return;
      }

      if (row.image_exists) {
        const src = `/image?path=${encodeURIComponent(row.image_path)}`;
        frame.innerHTML = `<img src="${src}" alt="${row.image_name}">`;
      } else {
        frame.innerHTML = `<div class="image-empty">找不到圖片檔案<br>${row.image_name}</div>`;
      }

      document.getElementById("detailYaw").textContent = fmt(row.yaw);
      document.getElementById("detailPitch").textContent = fmt(row.pitch);
      document.getElementById("detailStatus").textContent = row.status;
      document.getElementById("detailPerson").textContent = row.person_count;
      document.getElementById("detailExists").textContent = row.image_exists ? "Yes" : "No";
      document.getElementById("detailPath").textContent = row.image_path;
    }

    function selectRelative(offset) {
      if (!state.filtered.length) return;
      const currentIndex = Math.max(0, state.filtered.findIndex((row) => row.id === state.selectedId));
      let nextIndex = currentIndex + offset;
      if (nextIndex < 0) nextIndex = state.filtered.length - 1;
      if (nextIndex >= state.filtered.length) nextIndex = 0;
      state.selectedId = state.filtered[nextIndex].id;
      renderDetails(state.filtered[nextIndex]);
      drawPlot();
    }

    function bindEvents() {
      statusFilter.addEventListener("change", applyFilters);
      personFilter.addEventListener("change", applyFilters);
      searchInput.addEventListener("input", applyFilters);
      resetBtn.addEventListener("click", () => {
        statusFilter.value = "";
        personFilter.value = "";
        searchInput.value = "";
        applyFilters();
      });

      prevBtn.addEventListener("click", () => selectRelative(-1));
      nextBtn.addEventListener("click", () => selectRelative(1));
      openBtn.addEventListener("click", () => {
        const row = getSelectedRow();
        if (row && row.image_exists) {
          window.open(`/image?path=${encodeURIComponent(row.image_path)}`, "_blank", "noopener");
        }
      });

      canvas.addEventListener("mousemove", (event) => {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const hit = state.plotPoints.find((point) => Math.hypot(point.x - x, point.y - y) <= point.radius);
        const nextHover = hit ? hit.id : null;
        if (nextHover !== state.hoveredId) {
          state.hoveredId = nextHover;
          drawPlot();
        }
      });

      canvas.addEventListener("mouseleave", () => {
        if (state.hoveredId != null) {
          state.hoveredId = null;
          drawPlot();
        }
      });

      canvas.addEventListener("click", () => {
        if (state.hoveredId == null) return;
        state.selectedId = state.hoveredId;
        renderDetails(getSelectedRow());
        drawPlot();
      });

      canvas.addEventListener("wheel", (event) => {
        event.preventDefault();
        selectRelative(event.deltaY > 0 ? 1 : -1);
      }, { passive: false });

      window.addEventListener("keydown", (event) => {
        if (event.target && ["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
        if (event.key === "]" || event.key === "ArrowRight") selectRelative(1);
        if (event.key === "[" || event.key === "ArrowLeft") selectRelative(-1);
        if (event.key.toLowerCase() === "f" && state.filtered.length) {
          state.selectedId = state.filtered[0].id;
          renderDetails(state.filtered[0]);
          drawPlot();
        }
        if (event.key.toLowerCase() === "l" && state.filtered.length) {
          const row = state.filtered[state.filtered.length - 1];
          state.selectedId = row.id;
          renderDetails(row);
          drawPlot();
        }
      });

      window.addEventListener("resize", drawPlot);
    }

    boot().catch((error) => {
      document.body.innerHTML = `<pre style="padding:24px;">載入失敗\\n${error.stack || error}</pre>`;
    });
  </script>
</body>
</html>
"""


class ViewerHandler(BaseHTTPRequestHandler):
    payload: dict = {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html(HTML)
            return
        if parsed.path == "/api/data":
            self.send_json(self.payload)
            return
        if parsed.path == "/image":
            self.serve_image(parsed.query)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def serve_image(self, query: str) -> None:
        params = parse_qs(query)
        image_path = unquote((params.get("path") or [""])[0])
        if not image_path:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing image path")
            return

        target = Path(image_path)
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
            return

        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(target.stat().st_size))
        self.end_headers()
        with target.open("rb") as handle:
            self.wfile.write(handle.read())

    def send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_json(self, payload: dict) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, fmt: str, *args) -> None:
        return


def parse_args() -> argparse.Namespace:
    default_csv = Path("/media/ee303/4TB/sam3-body/sam-3d-body/deepfahsion.csv")
    parser = argparse.ArgumentParser(description="Inspect yaw / pitch labels against images.")
    parser.add_argument("--csv", type=Path, default=default_csv, help="CSV file to load")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8123, help="Port to bind")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    ViewerHandler.payload = build_payload(rows)
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    print(f"Loaded {len(rows)} rows from {args.csv}")
    print(f"Open http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
