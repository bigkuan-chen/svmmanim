import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.data_generator import generate_ring_dataset
from utils.svm_utils import train_svm, make_decision_grid, compute_decision_surface

# Page configuration
st.set_page_config(
    page_title="SVM Kernel Trick 3D Interactive Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling for a dark-mode, clean look with high-contrast text
st.markdown("""
<style>
    .main {
        background-color: #121212;
        color: #ffffff;
    }
    .stApp {
        background-color: #121212;
    }
    h1, h2, h3 {
        color: #ffd32a !important;
        font-family: 'Inter', sans-serif;
    }
    /* Make all markdown text, paragraphs, and lists highly legible white */
    .stMarkdown p, .stMarkdown li, .stMarkdown span, p, li {
        color: #f5f6fa !important;
        font-size: 15px;
    }
    /* Make sidebar parameters and labels clear */
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] p {
        color: #f5f6fa !important;
        font-size: 15px;
        font-weight: 500;
    }
    .stSlider > div > div > div > div {
        background-color: #ffd32a !important;
    }
    .reportview-container .markdown-text-container {
        font-family: 'Inter', sans-serif;
    }
    .metric-container {
        background-color: #1a1a1a;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333333;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #ffd32a;
    }
    .metric-label {
        font-size: 14px;
        color: #f5f6fa !important;
    }
    .note-container {
        background-color: #1e1e1d;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #ffd32a;
        margin-top: 20px;
    }
    .note-container p {
        color: #f5f6fa !important;
    }
</style>
""", unsafe_allow_html=True)

# Cache data generation for performance
@st.cache_data
def get_cached_dataset(n_inner, n_outer, noise, random_seed):
    return generate_ring_dataset(
        n_inner=n_inner,
        n_outer=n_outer,
        noise=noise,
        random_seed=random_seed
    )

# App Title & Description
st.title("SVM 核心技巧與 3D 決策曲面互動演示")
st.subheader("Support Vector Machine (SVM) Kernel Trick 3D Interactive Demo")

st.markdown("""
本互動教學工具旨在展示 **支援向量機 (SVM)** 如何利用**核技巧 (Kernel Trick)** 來解決 2D 空間中的非線性分類問題。
您可以透過調整左側邊欄的參數，即時觀察決策邊界（2D 投影）與模型信心曲面（3D 空間）的變化。
""")

# Sidebar Controls
st.sidebar.header("模型與數據參數控制")

# 1. Dataset controls
st.sidebar.subheader("1. 數據集設定")
number_of_points = st.sidebar.slider("樣本點總數", min_value=40, max_value=300, value=120, step=10)
noise = st.sidebar.slider("雜訊強度 (Noise)", min_value=0.0, max_value=0.5, value=0.08, step=0.01)
random_seed = st.sidebar.number_input("隨機種子 (Random Seed)", min_value=0, max_value=9999, value=7)

# Proportional division of points to keep the ring-like shape consistent
n_inner = int(number_of_points * 35 // 80)
n_outer = number_of_points - n_inner

# Generate cached data
X, y = get_cached_dataset(n_inner, n_outer, noise, random_seed)

# 2. SVM hyperparameters
st.sidebar.subheader("2. SVM 超參數設定")
kernel = st.sidebar.selectbox(
    "核函數類型 (Kernel)",
    options=["linear", "poly", "rbf", "sigmoid"],
    index=2  # Default to RBF
)

C = st.sidebar.slider(
    "正則化參數 C (C)",
    min_value=0.1,
    max_value=100.0,
    value=10.0,
    step=0.1,
    help="C 越大，對分類錯誤的容忍度越低（Margin 越窄）；C 越小，容忍度越高（Margin 越寬）"
)

# Conditional display of gamma
gamma = 1.0
if kernel in ["rbf", "poly", "sigmoid"]:
    gamma_option = st.sidebar.radio("Gamma 模式", ["scale", "auto", "手動設定"], index=2)
    if gamma_option == "手動設定":
        gamma = st.sidebar.slider(
            "核函數係數 Gamma (γ)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01,
            help="Gamma 越大，每個樣本影響力越集中，決策邊界越複雜；反之越平滑"
        )
    else:
        gamma = gamma_option.lower()

# Conditional display of degree
degree = 3
if kernel == "poly":
    degree = st.sidebar.slider(
        "多項式次數 (Degree)",
        min_value=2,
        max_value=6,
        value=3,
        step=1,
        help="僅在多項式核 (poly) 時生效"
    )

# Train the SVM Model
model = train_svm(X, y, kernel=kernel, C=C, gamma=gamma, degree=degree)

# Calculate metrics
y_pred = model.predict(X)
acc = np.mean(y_pred == y) * 100
n_sv = len(model.support_)
n_sv_class0 = np.sum(y[model.support_] == 0)
n_sv_class1 = np.sum(y[model.support_] == 1)

# Main Grid Tabs
tab1, tab2 = st.tabs(["📊 互動視覺化圖表", "📚 教學筆記與概念說明"])

with tab1:
    # Top Row: Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{acc:.1f}%</div>
            <div class="metric-label">訓練集準確率 (Accuracy)</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{n_sv} <span style="font-size:14px;color:#888;">({n_sv/number_of_points*100:.1f}%)</span></div>
            <div class="metric-label">支援向量總數 (Support Vectors)</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{n_sv_class0} / {n_sv_class1}</div>
            <div class="metric-label">支援向量分佈 (內圈 / 外圈)</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="text-transform: uppercase;">{kernel}</div>
            <div class="metric-label">當前核函數 (Kernel)</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Insert video player for Phase 1 inside Tab 1 ---
    import os
    st.markdown("### 🎬 Phase 1 概念動畫：從 2D 到 3D 的特徵映射")
    video_paths = [
        "media/videos/phase1_manim_kernel_trick/1080p60/SVMKernelTrick3D.mp4",
        "media/videos/phase1_manim_kernel_trick/480p15/SVMKernelTrick3D.mp4"
    ]
    video_file = None
    for path in video_paths:
        if os.path.exists(path):
            video_file = path
            break
            
    if video_file:
        st.video(video_file)
    else:
        st.info("💡 提示：在本地端執行 `manim -pql phase1_manim_kernel_trick.py SVMKernelTrick3D` 生成動畫後，影片將會在此處顯示。")

    # Compute grid for contour and surface plotting
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy, grid_points = make_decision_grid(
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        resolution=80  # Optimized for rendering speed in web
    )
    Z = compute_decision_surface(model, grid_points, grid_shape=xx.shape)

    # 2D Plot
    st.subheader("2D 原始空間與決策邊界 Projection")
    
    fig_2d = go.Figure()
    
    # 1. Add Decision Boundary Contour (f = 0)
    fig_2d.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 80),
        y=np.linspace(y_min, y_max, 80),
        z=Z,
        contours_coloring='none',
        line=dict(width=3, color='#ffd32a'),
        contours=dict(start=0, end=0, size=1),
        showscale=False,
        name='決策邊界 (f=0)',
        hoverinfo='skip'
    ))
    
    # 2. Add Margin Contours (f = -1 and +1)
    fig_2d.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 80),
        y=np.linspace(y_min, y_max, 80),
        z=Z,
        contours_coloring='none',
        line=dict(width=1.5, color='rgba(255,255,255,0.4)', dash='dash'),
        contours=dict(start=-1, end=1, size=2),
        showscale=False,
        name='邊界邊界 (f=±1)',
        hoverinfo='skip'
    ))
    
    # 3. Scatter Class 0 (Inner)
    fig_2d.add_trace(go.Scatter(
        x=X[y==0, 0], y=X[y==0, 1],
        mode='markers',
        marker=dict(size=8, color='#1e90ff', line=dict(color='#121212', width=0.8)),
        name='類別 0 (內圈藍點)'
    ))
    
    # 4. Scatter Class 1 (Outer)
    fig_2d.add_trace(go.Scatter(
        x=X[y==1, 0], y=X[y==1, 1],
        mode='markers',
        marker=dict(size=8, color='#ff4757', line=dict(color='#121212', width=0.8)),
        name='類別 1 (外圈紅點)'
    ))
    
    # 5. Highlight Support Vectors
    sv_indices = model.support_
    fig_2d.add_trace(go.Scatter(
        x=X[sv_indices, 0], y=X[sv_indices, 1],
        mode='markers',
        marker=dict(size=14, color='rgba(0,0,0,0)', line=dict(color='#ffd32a', width=1.5)),
        name='支援向量 (Support Vectors)'
    ))
    
    fig_2d.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#121212',
        font_color='#ffffff',
        xaxis=dict(gridcolor='#333333', zeroline=False, title="特徵 X1"),
        yaxis=dict(gridcolor='#333333', zeroline=False, title="特徵 X2", scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450
    )
    
    st.plotly_chart(fig_2d, width='stretch')

    # 3D Plot
    st.subheader("3D 信心函數決策曲面 z = f(X1, X2)")
    
    fig_3d = go.Figure()
    
    # 1. Add Decision Surface z = f(x, y)
    fig_3d.add_trace(go.Surface(
        x=np.linspace(x_min, x_max, 80),
        y=np.linspace(y_min, y_max, 80),
        z=Z,
        colorscale='RdBu',
        opacity=0.75,
        showscale=True,
        colorbar=dict(
            title=dict(text='f(X1, X2) 分數', font=dict(color='white')),
            thickness=15,
            len=0.7,
            tickcolor='white',
            tickfont=dict(color='white')
        ),
        name='決策信心曲面'
    ))
    
    # 2. Add z = 0 reference plane
    fig_3d.add_trace(go.Surface(
        x=np.linspace(x_min, x_max, 80),
        y=np.linspace(y_min, y_max, 80),
        z=np.zeros_like(Z),
        colorscale=[[0, 'rgba(255, 211, 42, 0.1)'], [1, 'rgba(255, 211, 42, 0.1)']],
        showscale=False,
        opacity=0.25,
        hoverinfo='skip',
        name='z = 0 基準平面'
    ))
    
    # Compute decision scores for training data points to plot them in 3D
    Z_points = model.decision_function(X)
    
    # 3. Highlight Support Vectors in 3D (Golden background markers to create a halo effect)
    fig_3d.add_trace(go.Scatter3d(
        x=X[sv_indices, 0], y=X[sv_indices, 1], z=Z_points[sv_indices],
        mode='markers',
        marker=dict(size=10, color='#ffd32a', line=dict(color='#121212', width=0.5)),
        name='支援向量 (Support Vectors)'
    ))
    
    # 4. Class 0 points in 3D
    fig_3d.add_trace(go.Scatter3d(
        x=X[y==0, 0], y=X[y==0, 1], z=Z_points[y==0],
        mode='markers',
        marker=dict(size=6, color='#1e90ff', line=dict(color='#121212', width=0.5)),
        name='類別 0 (內圈藍點)'
    ))
    
    # 5. Class 1 points in 3D
    fig_3d.add_trace(go.Scatter3d(
        x=X[y==1, 0], y=X[y==1, 1], z=Z_points[y==1],
        mode='markers',
        marker=dict(size=6, color='#ff4757', line=dict(color='#121212', width=0.5)),
        name='類別 1 (外圈紅點)'
    ))
    
    fig_3d.update_layout(
        paper_bgcolor='#121212',
        font_color='#ffffff',
        scene=dict(
            xaxis=dict(backgroundcolor='#121212', gridcolor='#333333', showbackground=True, title="特徵 X1"),
            yaxis=dict(backgroundcolor='#121212', gridcolor='#333333', showbackground=True, title="特徵 X2"),
            zaxis=dict(backgroundcolor='#121212', gridcolor='#333333', showbackground=True, title="f(X1, X2) 分數"),
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    st.plotly_chart(fig_3d, width='stretch')

    # Bottom Section: Dynamic Parameter Commentary / Teaching Notes
    st.markdown("### 💡 當前參數特徵與教學指南")
    
    # Analyze parameters and give advice
    advice = []
    
    # Kernel specific advice
    if kernel == "linear":
        advice.append("⚠️ **線性核函數 (Linear Kernel)**: 決策邊界是一條直線。此時模型無法有效分割環形分佈的資料，即使調整 C 也無法改善非線性分類。")
    elif kernel == "poly":
        advice.append(f"🌀 **多項式核函數 (Polynomial Kernel - 次數: {degree})**: 透過多項式基底投射。次數越高，邊界的複雜度越高。請注意高次多項式在邊緣容易出現極端的發散情況。")
    elif kernel == "rbf":
        advice.append("✨ **徑向基底核函數 (RBF Kernel)**: 經典的非線性核。藉由將資料映射到無限維的特徵空間，能夠極度自然地生成環狀的決策邊界。")
    elif kernel == "sigmoid":
        advice.append("⚡ **Sigmoid 核函數**: 與神經網絡的多層感知器類似。在非線性環形數據上，它的行為通常不夠穩定，決策面可能會出現扭曲。")

    # C advice
    if C < 1.0:
        advice.append(f"🟢 **C值較小 (C={C})**: 代表模型容忍較多的分類錯誤，以換取較寬的間距 (Soft Margin)。決策曲面通常會被拉平，防範過擬合的能力較好。")
    elif C > 20.0:
        advice.append(f"🔴 **C值較大 (C={C})**: 代表模型極度不願意看到錯誤分類，迫使間距縮窄 (Hard Margin)。此時支援向量點會變少，曲面可能因追逐噪聲點而過度抖動。")

    # Gamma advice
    if isinstance(gamma, float):
        if gamma < 0.2:
            advice.append(f"🟢 **Gamma值較小 (γ={gamma})**: 每個支援向量的影響範圍很廣（高 Bias，低 Variance）。信心曲面會非常平緩，決策邊界會更圓潤、平滑。")
        elif gamma > 3.0:
            advice.append(f"🔴 **Gamma值較大 (γ={gamma})**: 每個支援向量的影響範圍非常狹窄、尖銳（低 Bias，高 Variance）。信心曲面會呈現如山峰般的急起急落，容易產生過擬合現象（Overfitting）。")

    # Display notes in a stylized container
    notes_html = "".join([f"<p style='margin-bottom:10px;'>{item}</p>" for item in advice])
    st.markdown(f"""
    <div class="note-container">
        {notes_html}
    </div>
    """, unsafe_allow_html=True)


with tab2:
    st.markdown(r"""
    ## 📘 核技巧 (Kernel Trick) 核心教學筆記
    
    ### 1. 為什麼需要核技巧？
    在機器學習中，許多現實世界的資料並非**線性可分 (Linearly Separable)**。例如我們所使用的環形資料集：
    - **藍色點 (類別 0)**：分佈在圓心附近。
    - **紅色點 (類別 1)**：分佈在外圍圓環。
    
    在 2D 原始空間中，我們絕對找不到任何一條**直線** $w^T x + b = 0$ 能將這兩組點完美分開。
    
    ### 2. 映射至高維特徵空間
    解決非線性問題的標準做法，是定義一個特徵映射函數 $\phi(x)$，將資料從低維投影到高維。
    
    在 Phase 1 的動畫中，我們使用了一個直觀的映射：
    $$\phi(x, y) = (x, y, x^2 + y^2)$$
    
    這相當於給每個點加上了一個高度 $z = x^2 + y^2$（即該點到原點的距離平方）。
    - 內圈藍色點離原點近 $\rightarrow z$ 值較小，保持在下方。
    - 外圈紅色點離原點遠 $\rightarrow z$ 值較大，被拉往上方。
    
    此時，在 3D 特徵空間中，我們便可以使用一個**水平的超平面** $z = 1.8$ 將上下兩群點分開！
    """)

    # --- Insert video player for Phase 1 ---
    import os
    st.markdown("#### 🎬 Phase 1: SVM 3D 投影概念動畫影片")
    video_paths = [
        "media/videos/phase1_manim_kernel_trick/1080p60/SVMKernelTrick3D.mp4",
        "media/videos/phase1_manim_kernel_trick/480p15/SVMKernelTrick3D.mp4"
    ]
    video_file = None
    for path in video_paths:
        if os.path.exists(path):
            video_file = path
            break
            
    if video_file:
        st.video(video_file)
    else:
        st.info("💡 提示：在本地端執行 `manim -pql phase1_manim_kernel_trick.py SVMKernelTrick3D` 生成動畫後，影片將會在此處顯示。")

    st.markdown(r"""
    ### 3. 超平面的 2D 投影
    當我們將這個 3D 空間的超平面 $z = c$ 投影回 2D 原始空間時：
    $$x^2 + y^2 = c$$
    這恰好就是一個**圓形決策邊界**。這解釋了為什麼在 3D 空間的線性分割，在 2D 空間中呈現的是圓形的非線性分割。
    
    ---
    
    ### 🔬 重要的數學澄清：RBF 核的本質
    
    > [!IMPORTANT]
    > **RBF 核函數 (徑向基底核 / Gaussian Kernel)** 
    >
    > $$K(x, x') = \exp(-\gamma ||x - x'||^2)$$
    > 
    > **並非**簡單地將資料映射到 3D。事實上，RBF 核函數隱式地將資料映射到一個**無限維的特徵空間**！
    > 
    > 我們在此互動程式中畫出的 3D 曲面，其高度 $z$ 代表的是 SVM 訓練出來的**決策信心函數值 (Decision Function)** $f(x)$：
    > 
    > $$f(x) = \sum_{i \in \text{SV}} \alpha_i y_i K(x_i, x) + b$$
    > 
    > 曲面在 $z=0$ 的黃色截線即為決策邊界。這是一個極為重要的概念，避免學生誤以為 RBF 核只是做了 $z = x^2 + y^2$ 的映射。
    
    ### 🎯 教學思考問題
    1. **調整雜訊 (Noise)** 至 `0.2` 以上，將 `C` 設為 `100.0`，並將 `Gamma` 設為 `5.0`。觀察 2D 決策邊界與 3D 信心曲面。
       - *思考*：此時的 3D 曲面是不是出現了許多小山丘和谷地？這代表模型發生了什麼現象？（解答：過擬合，模型過度適應個別噪聲點）。
    2. **切換為線性核 (Linear Kernel)**。不論你如何調整 `C`，模型是否還能分開這組資料？
       - *思考*：為什麼 3D 信心曲面現在變成了一個完全平坦的傾斜面？（解答：因為線性核的決策信心函數是線性特徵的線性組合，其圖形必然是一個傾斜的平面）。
    """)
