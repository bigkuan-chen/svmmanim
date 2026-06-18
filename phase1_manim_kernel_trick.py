import numpy as np
from manim import *
from utils.data_generator import generate_ring_dataset

class SVMKernelTrick3D(ThreeDScene):
    def construct(self):
        # 1. Opening Title
        title = Text("SVM Kernel Trick: From 2D to 3D", font_size=36, color=YELLOW)
        title.to_edge(UP)
        subtitle = Text("無法用直線在 2D 空間分割 -> 提升至 3D 特徵空間", font_size=24)
        subtitle.next_to(title, DOWN, buff=0.3)
        
        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.play(FadeIn(title), Write(subtitle))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))
        
        # Set up 3D Axes
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 8, 2],
            x_length=6,
            y_length=6,
            z_length=4,
        )
        # Start in 2D orthographic-like view looking down the Z-axis
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES)
        
        self.play(FadeIn(axes))
        
        # 2. Generate and show 2D data (z = 0)
        X, y = generate_ring_dataset(n_inner=35, n_outer=45, noise=0.0, random_seed=7)
        
        dots = VGroup()
        for point, label in zip(X, y):
            x_coord, y_coord = point[0], point[1]
            color = BLUE if label == 0 else RED
            dot = Dot3D(point=axes.c2p(x_coord, y_coord, 0), color=color, radius=0.08)
            dots.add(dot)
            
        # Text for 2D separation limitation
        text_2d = Text("在 2D 原始空間中，藍點與紅點是非線性不可分的（無法用直線分割）", font_size=18)
        text_2d.to_edge(UP)
        self.add_fixed_in_frame_mobjects(text_2d)
        
        self.play(FadeIn(dots), FadeIn(text_2d))
        self.wait(3)
        self.play(FadeOut(text_2d))
        
        # 3. Show mapping formula
        formula = Text("φ(x, y) = (x, y, x² + y²)", font_size=20, color=WHITE)
        formula.to_edge(UP)
        self.add_fixed_in_frame_mobjects(formula)
        self.play(Write(formula))
        self.wait(2)
        
        # 4. Move camera to 3D perspective
        self.move_camera(phi=65 * DEGREES, theta=-45 * DEGREES, run_time=2)
        self.wait(1)
        
        # 5. Animate lifting points to 3D according to z = x^2 + y^2
        lift_animations = []
        for dot, point in zip(dots, X):
            x_coord, y_coord = point[0], point[1]
            z_coord = x_coord**2 + y_coord**2
            lift_animations.append(dot.animate.move_to(axes.c2p(x_coord, y_coord, z_coord)))
            
        self.play(*lift_animations, run_time=3)
        self.wait(1.5)
        
        # 6. Show the paraboloid surface z = x^2 + y^2
        paraboloid = Surface(
            lambda u, v: axes.c2p(u, v, u**2 + v**2),
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            fill_opacity=0.22,
            checkerboard_colors=[BLUE_D, BLUE_E],
        )
        self.play(Create(paraboloid), run_time=2)
        self.wait(1.5)
        
        # 7. Show the separating hyperplane at z = 1.8
        # Since inner points have z <= 1.0 and outer points have z >= 2.56
        hyperplane = Surface(
            lambda u, v: axes.c2p(u, v, 1.8),
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            fill_opacity=0.35,
            color=YELLOW,
        )
        
        plane_label = Text("特徵空間中的線性超平面 (z = 1.8)", font_size=18, color=YELLOW)
        plane_label.to_edge(RIGHT)
        self.add_fixed_in_frame_mobjects(plane_label)
        
        self.play(Create(hyperplane), FadeIn(plane_label))
        self.wait(2)
        self.play(FadeOut(plane_label))
        
        # 8. Project back to 2D plane (z = 0)
        # x^2 + y^2 = 1.8 implies a circle with radius = sqrt(1.8) ≈ 1.34
        r_decision = np.sqrt(1.8)
        decision_circle = ParametricFunction(
            lambda t: axes.c2p(r_decision * np.cos(t), r_decision * np.sin(t), 0),
            t_range=[0, 2 * np.pi],
            color=YELLOW,
            stroke_width=4,
        )
        
        circle_label = Text("超平面投影回 2D 空間，形成圓形的非線性決策邊界", font_size=18, color=YELLOW)
        circle_label.to_edge(UP)
        self.add_fixed_in_frame_mobjects(circle_label)
        
        self.play(Create(decision_circle), FadeIn(circle_label))
        self.wait(3)
        self.play(FadeOut(circle_label), FadeOut(formula))
        
        # 9. Camera rotation to show spatial relationships clearly
        self.begin_ambient_camera_rotation(rate=0.18)
        self.wait(5)
        self.stop_ambient_camera_rotation()
        
        # 10. Final Summary View
        summary_title = Text("總結", font_size=28, color=YELLOW).to_edge(UP)
        summary1 = Text("1. 3D 特徵空間中：資料可用線性超平面完美分割 (Linear Hyperplane)", font_size=18).next_to(summary_title, DOWN, buff=0.3)
        summary2 = Text("2. 2D 原始空間中：超平面投影回 2D 呈現圓形非線性邊界 (Nonlinear Boundary)", font_size=18).next_to(summary1, DOWN, buff=0.2)
        summary3 = Text("3. 這就是「核技巧 (Kernel Trick)」將低維不可分提升至高維可分的直觀概念！", font_size=18).next_to(summary2, DOWN, buff=0.2)
        
        summary_group = VGroup(summary_title, summary1, summary2, summary3)
        self.add_fixed_in_frame_mobjects(summary_group)
        self.play(FadeIn(summary_group))
        self.wait(4)
        self.play(
            FadeOut(summary_group),
            FadeOut(dots),
            FadeOut(axes),
            FadeOut(paraboloid),
            FadeOut(hyperplane),
            FadeOut(decision_circle)
        )
        self.wait(1)
