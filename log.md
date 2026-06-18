# SVM Kernel Trick 3D Interactive Demo - 開發紀錄與討論日誌 (Development Log)

本文件記錄了本專案在開發過程中的所有技術決策、問題解決方案，以及與使用者的 Q&A 對話紀錄。

---

## 1. 專案開發歷程與實作細節

### Phase 1: Manim 概念動畫
* **檔案**：`phase1_manim_kernel_trick.py`
* **設計要點**：
  * 使用 `ThreeDScene` 來展示低維資料到高維特徵空間的轉換。
  * 初始視角設定為 $z$ 軸正上方的 2D 正交俯視圖，直觀展示 2D 環狀資料（內圈藍點，外圈紅點）在 2D 原始空間中線性不可分的限制。
  * 移動相機視角至 3D 傾斜角度（$\phi=65^\circ, \theta=-45^\circ$），並將所有點沿著 $z = x^2 + y^2$ 進行動態提升。
  * 繪製半透明藍色拋物面（Paraboloid Surface）與黃色水平分割超平面（Hyperplane, $z = 1.8$），展示高維空間中的線性可分性。
  * 將超平面與拋物面的交線投影回 $z=0$ 平面，畫出圓形的黃色決策邊界，展現核技巧的直觀幾何解釋。
  * 所有公式（MathTex）與文字說明使用 `add_fixed_in_frame_mobjects` 鎖定在 HUD 畫面上，避免隨相機旋轉。

### Phase 2: 真實 RBF SVM 決策曲面
* **檔案**：`phase2_rbf_decision_surface.py`
* **設計要點**：
  * 使用 `scikit-learn` 訓練一個真實的 RBF SVM（$C=10, \gamma=1$）。
  * 以 Matplotlib 畫出精美的深色調（Dark Mode）對比圖。
  * **2D 投影圖**：繪製決策線 $f(x,y)=0$、間距線 $f(x,y)=\pm 1$，並以黃色空心圓環高亮標出支援向量。
  * **3D 信心曲面圖**：將 $z$ 軸設為模型決策函數值 $f(x, y)$，藍點與紅點散佈在各自的決策信心高度上，能清晰看出內圈凹陷成藍色山谷、外圈隆起為紅色高地的特徵。

### Phase 3: Streamlit 與 Plotly 互動式網頁
* **檔案**：`phase3_streamlit_app.py`
* **設計要點**：
  * 使用 Streamlit 建立網頁架構，左側邊欄可調整 Kernel、C、Gamma、Degree、Noise 等參數。
  * 使用 Plotly 繪製互動式 2D/3D 圖表，支援滑鼠懸停、旋轉與縮放。
  * 內建動態教學提示指南（對應不同的超參數變化給予即時回饋）與詳細的數學概念澄清（解釋 RBF 核其實映射至無限維度，3D 繪製的是決策函數曲面）。

---

## 2. 遭遇問題與解決方案 (Bug Fixes)

### 📌 問題 A: Manim 安裝與編譯環境衝突
* **起因**：本地系統使用的是 Python 3.14 預覽版，PyPI 上尚未有針對 Python 3.14 的 `moderngl` 與 `glcontext` 預編譯二進位 Wheel，導致 pip 嘗試從原始碼編譯，而在缺少 Microsoft C++ Build Tools 的系統上建置失敗。
* **解決方案**：
  * 先將除了 `manim` 以外的其餘核心套件（`numpy`, `scikit-learn`, `matplotlib`, `streamlit`, `plotly`, `pandas`）進行安裝，確保 Phase 2 與 Phase 3 能順利在本地執行。
  * 在 [README.md](file:///d:/cena/0618/HW8/README.md) 中添加了詳細說明，告知使用者 Manim 需要額外配置 C++ 編譯環境與 LaTeX/FFmpeg 才能渲染 Phase 1 影片。

### 📌 問題 B: Plotly 屬性錯誤 (Bad Property Path: titlefont)
* **起因**：在 `phase3_streamlit_app.py` 中，我們在 Plotly 3D 曲面的 `colorbar` 設定中使用了 `titlefont=dict(color='white')`。在新版 Plotly 中，此屬性已被移入 `title` 子字典中。
* **解決方案**：
  * 將該屬性修正為：`title=dict(text='f(X1, X2) 分數', font=dict(color='white'))`，修復了 Streamlit 啟動時的報錯。

### 📌 問題 C: Streamlit 使用容器寬度警告 (use_container_width Deprecation)
* **起因**：原程式碼使用了舊版的 `st.plotly_chart(..., use_container_width=True)`，控制台跳出即將棄用的警告。
* **解決方案**：
  * 將其更改為新版的參數：`st.plotly_chart(..., width='stretch')`，移除了所有警告訊息。

---

## 3. 重要 Q&A 與界面優化討論

### 💬 Q1: 網頁畫面排版調整
* **使用者提問**：*「網頁畫面，將3d圖形放至2d圖形下方」*
* **調整內容**：
  * 我們移除了原本左右並排的兩欄佈局（`col1, col2 = st.columns(2)`），讓 2D 投影圖與 3D 信心曲面圖垂直排列。
  * 垂直排列能提供更寬敞的橫向空間，讓學生能更細緻地旋轉與縮放 3D 信心曲面。

### 💬 Q2: 2D 圓形拉伸與高度縮小問題
* **使用者提問**：*「2d圖比例不對，太寬，也縮小一點高度」*
* **調整內容**：
  * **比例修正**：由於改為單欄垂直排列後，寬度會被拉滿，導致原本的正圓形資料集被橫向拉扁成橢圓形。我們在 2D 圖的 y 軸設定中加上了 `scaleanchor="x"` 與 `scaleratio=1`，強制 x 軸與 y 軸維持 1:1 的幾何比例，確保圓形永遠是正圓。
  * **高度調整**：在 `update_layout` 中將 2D 圖的高度限制為 `450` 像素，3D 圖限制為 `500` 像素，防止垂直堆疊時版面過長，提升了排版的緊湊度。

### 💬 Q3: 為什麼 3D 圖多了黃色點，2D 圖只有紅色與藍色點？
* **使用者提問**：*「為什麼3d圖多了黃色點，2d圖只有紅色與藍色點」*
* **原理解釋**：
  * 2D 圖中的支援向量是用 **「空心黃色圓圈」** 標示，它疊加在原本的紅色/藍色資料點外圍。
  * 3D 圖（WebGL）由於技術限制，原本是直接繪製 **「實心黃色圓點」** 來標記，因此在視覺上黃色實心點會徹底遮住底下的紅點/藍點，看起來就像是獨立多出來的黃色數據點。
* **調整內容（Halo 暈光效果）**：
  * 我們調整了 3D 散佈圖的**繪製順序與點的尺寸**：
    1. 首先繪製**較大的黃色支援向量點（大小為 10）**作為背景底圖。
    2. 接著在上方繪製**正常大小的紅色與藍色點（大小為 6）**。
  * 如此一來，紅點/藍點會精準壓在黃點的中央，使得邊緣露出一圈黃色的外環，在 3D 空間中完美模擬了「空心黃色圈」的標記效果，視覺語義與 2D 圖達到完全統一。
