import os
import numpy as np
import matplotlib.pyplot as plt
from utils.data_generator import generate_ring_dataset
from utils.svm_utils import train_svm, make_decision_grid, compute_decision_surface

def main():
    # Set dark theme for a premium, modern visual style
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 7), facecolor='#121212')
    
    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # 2. Generate the circular dataset with default noise
    X, y = generate_ring_dataset(n_inner=35, n_outer=45, noise=0.08, random_seed=7)
    
    # 3. Train the real RBF SVM classifier
    C = 10.0
    gamma = 1.0
    model = train_svm(X, y, kernel='rbf', C=C, gamma=gamma)
    
    # Calculate model metrics
    y_pred = model.predict(X)
    acc = np.mean(y_pred == y) * 100
    n_sv = len(model.support_)
    n_sv_class0 = np.sum(y[model.support_] == 0)
    n_sv_class1 = np.sum(y[model.support_] == 1)
    
    print("==================================================")
    print("Real RBF SVM Model Summary")
    print("--------------------------------------------------")
    print(f"Kernel: RBF | C: {C} | Gamma: {gamma}")
    print(f"Training Accuracy: {acc:.2f}%")
    print(f"Total Support Vectors: {n_sv}")
    print(f"  - Class 0 (Inner/Blue): {n_sv_class0}")
    print(f"  - Class 1 (Outer/Red):  {n_sv_class1}")
    print("==================================================")
    
    # 4. Create decision boundary evaluation grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy, grid_points = make_decision_grid(
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        resolution=100
    )
    
    # Evaluate decision scores on grid
    Z = compute_decision_surface(model, grid_points, grid_shape=xx.shape)
    
    # Define aesthetic colors matching the design specs
    color_inner = '#1e90ff'  # Dodson Blue for inner class
    color_outer = '#ff4757'  # Watermelon Red for outer class
    color_sv = '#ffd32a'     # Bright Yellow/Gold for boundaries and support vectors
    
    point_colors = np.where(y == 0, color_inner, color_outer)
    
    # ================= SUBPLOT 1: 2D Decision Boundary =================
    ax2d = fig.add_subplot(1, 2, 1)
    ax2d.set_facecolor('#1a1a1a')
    
    # Draw decision boundary (f(x, y) = 0) and margins (f(x, y) = -1, +1)
    contour_db = ax2d.contour(xx, yy, Z, levels=[0], colors=[color_sv], linewidths=[3.0], zorder=3)
    contour_margin = ax2d.contour(
        xx, yy, Z,
        levels=[-1, 1],
        colors=['#888888'],
        linestyles=['dashed'],
        linewidths=[1.5],
        zorder=2
    )
    
    # Add label markers to contours
    ax2d.clabel(contour_db, inline=True, fmt={0: 'f(x,y)=0 (Boundary)'}, fontsize=10, colors=color_sv)
    ax2d.clabel(contour_margin, inline=True, fmt={-1: 'f(x,y)=-1', 1: 'f(x,y)=+1'}, fontsize=9, colors='#aaaaaa')
    
    # Scatter training data points
    ax2d.scatter(
        X[:, 0], X[:, 1],
        c=point_colors,
        s=50,
        edgecolors='#121212',
        linewidths=0.5,
        zorder=4
    )
    
    # Highlight support vectors with larger gold rings
    sv_idx = model.support_
    ax2d.scatter(
        X[sv_idx, 0], X[sv_idx, 1],
        s=130,
        facecolors='none',
        edgecolors=color_sv,
        linewidths=1.5,
        zorder=5,
        label='Support Vectors'
    )
    
    ax2d.set_title(f'2D SVM Decision Boundary & Margins\n(Accuracy: {acc:.1f}%)', fontsize=12, pad=15, color='#ffffff')
    ax2d.set_xlabel('Feature X1', fontsize=10, color='#cccccc')
    ax2d.set_ylabel('Feature X2', fontsize=10, color='#cccccc')
    ax2d.grid(color='#333333', linestyle=':', alpha=0.5)
    
    # Custom aesthetic legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_inner, markersize=8, label='Class 0 (Inner Blue)', markeredgecolor='#121212'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_outer, markersize=8, label='Class 1 (Outer Red)', markeredgecolor='#121212'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor=color_sv, markersize=10, markeredgewidth=1.5, label='Support Vectors'),
        Line2D([0], [0], color=color_sv, lw=2.5, label='Decision Boundary (f=0)'),
        Line2D([0], [0], color='#888888', lw=1.5, linestyle='--', label='Margin Boundary (f=±1)')
    ]
    ax2d.legend(handles=legend_elements, loc='upper right', framealpha=0.6, facecolor='#1a1a1a', edgecolor='#333333')
    
    # ================= SUBPLOT 2: 3D Decision Surface =================
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    ax3d.set_facecolor('#121212')
    
    # Plot Decision Function Surface: z = f(x, y)
    # RBF outputs represent confidence heights
    surf = ax3d.plot_surface(
        xx, yy, Z,
        cmap='coolwarm',
        alpha=0.75,
        edgecolor='none',
        rstride=1,
        cstride=1,
        zorder=1
    )
    
    # Add a translucent yellow grid at z = 0 reference plane
    ax3d.plot_surface(xx, yy, np.zeros_like(xx), color='#ffd32a', alpha=0.12, edgecolor='none', zorder=2)
    
    # Plot 3D training points positioned at their corresponding decision scores: z = f(x, y)
    Z_points = model.decision_function(X)
    ax3d.scatter(
        X[:, 0], X[:, 1], Z_points,
        c=point_colors,
        s=45,
        edgecolors='#121212',
        linewidths=0.5,
        depthshade=False,
        zorder=10
    )
    
    # Highlight support vectors in 3D space
    ax3d.scatter(
        X[sv_idx, 0], X[sv_idx, 1], Z_points[sv_idx],
        facecolors='none',
        edgecolors=color_sv,
        s=100,
        linewidths=1.8,
        depthshade=False,
        zorder=11
    )
    
    ax3d.set_title('3D SVM Decision Function Surface\nHeight z = f(X1, X2)', fontsize=12, pad=15, color='#ffffff')
    ax3d.set_xlabel('Feature X1', fontsize=10, color='#cccccc', labelpad=10)
    ax3d.set_ylabel('Feature X2', fontsize=10, color='#cccccc', labelpad=10)
    ax3d.set_zlabel('Decision Score f(X1, X2)', fontsize=10, color='#cccccc', labelpad=10)
    
    # Style the 3D axis panes for a premium integrated look
    ax3d.xaxis.set_pane_color((0.07, 0.07, 0.07, 1.0))
    ax3d.yaxis.set_pane_color((0.07, 0.07, 0.07, 1.0))
    ax3d.zaxis.set_pane_color((0.07, 0.07, 0.07, 1.0))
    
    # Adjust viewing perspective
    ax3d.view_init(elev=30, azim=-45)
    
    plt.tight_layout()
    
    # Save the output image
    output_path = 'outputs/rbf_decision_surface.png'
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Plot saved successfully to {output_path}")
    
    # Show plot with fallback for headless terminals
    try:
        plt.show()
    except Exception as e:
        print(f"\n[Note] Could not open interactive graphical window: {e}")
        print("This is normal in headless environments. The visualization has been saved as an image.")
        print(f"You can view it at: {os.path.abspath(output_path)}")

if __name__ == '__main__':
    main()
