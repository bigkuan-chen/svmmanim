project:
  name: "SVM Kernel Trick 3D Interactive Demo"
  version: "v1.0"
  language: "zh-TW + English code comments"
  goal: >
    Build a complete educational demonstration for Support Vector Machine kernel trick.
    The demo should include a Manim animation, a real sklearn RBF SVM decision surface,
    and an interactive Streamlit/Plotly app for students to explore kernel parameters.

  target_audience:
    - 高中資訊課學生
    - 大學初階機器學習學生
    - 教師教學簡報與課堂展示

  core_story:
    - "2D 中心藍點與外圈紅點無法被直線分開。"
    - "透過 feature mapping 將資料提升到 3D。"
    - "在 3D feature space 中，可以用 hyperplane 分開資料。"
    - "3D hyperplane 投影回 2D 後，形成非線性決策邊界。"
    - "進一步展示真正的 RBF SVM decision function surface。"
    - "最後提供互動式介面讓學生調整 C、gamma、kernel。"

  technical_stack:
    animation:
      - "Python"
      - "Manim Community Edition"
      - "NumPy"
    machine_learning:
      - "scikit-learn"
      - "NumPy"
      - "matplotlib"
    interactive_demo:
      - "Streamlit"
      - "Plotly"
      - "scikit-learn"
      - "pandas"

  repository_structure:
    expected_files:
      - "README.md"
      - "requirements.txt"
      - "phase1_manim_kernel_trick.py"
      - "phase2_rbf_decision_surface.py"
      - "phase3_streamlit_app.py"
      - "utils/data_generator.py"
      - "utils/svm_utils.py"
      - "assets/"
      - "outputs/"

  installation:
    requirements_txt:
      - "manim"
      - "numpy"
      - "scikit-learn"
      - "matplotlib"
      - "streamlit"
      - "plotly"
      - "pandas"

  commands:
    install:
      - "pip install -r requirements.txt"
    run_manim_preview:
      - "manim -pql phase1_manim_kernel_trick.py SVMKernelTrick3D"
    run_manim_high_quality:
      - "manim -pqh phase1_manim_kernel_trick.py SVMKernelTrick3D"
    run_rbf_surface:
      - "python phase2_rbf_decision_surface.py"
    run_streamlit:
      - "streamlit run phase3_streamlit_app.py"

phases:
  - phase_id: 1
    title: "Manim Concept Animation: Kernel Trick from 2D to 3D"
    objective: >
      Create a clean educational Manim animation showing how a circularly separable
      2D dataset becomes linearly separable after mapping to 3D using z = x^2 + y^2.

    output_file: "phase1_manim_kernel_trick.py"

    educational_message:
      main_idea: >
        In the original 2D space, the data cannot be separated by a straight line.
        After applying a feature mapping phi(x, y) = (x, y, x^2 + y^2),
        the blue inner points and red outer points become separable by a horizontal hyperplane.
      key_formulae:
        - "\\phi(x, y) = (x, y, x^2 + y^2)"
        - "z = x^2 + y^2"
        - "z = c"
        - "x^2 + y^2 = c"

    data_design:
      random_seed: 7
      blue_class:
        description: "Inner cluster near the origin."
        label: 0
        color: "BLUE"
        radius_range: [0.0, 1.0]
        n_points: 35
      red_class:
        description: "Outer circular ring."
        label: 1
        color: "RED"
        radius_range: [1.6, 2.5]
        n_points: 45

    animation_sequence:
      - step: 1
        name: "Opening title"
        description: >
          Show title: 'SVM Kernel Trick: From 2D to 3D'.
          Add brief subtitle: 'Nonlinear in 2D, linear in feature space.'
      - step: 2
        name: "Show 2D data"
        description: >
          Display blue points in the center and red points in the outer ring
          on the z = 0 plane. Add text: 'No straight line can separate them in 2D.'
      - step: 3
        name: "Show mapping formula"
        description: >
          Display MathTex formula phi(x, y) = (x, y, x^2 + y^2).
          Explain visually that each point is lifted upward according to its distance from origin.
      - step: 4
        name: "Animate lifting to 3D"
        description: >
          Transform each point from (x, y, 0) to (x, y, x^2 + y^2).
          Blue points remain low; red points move higher.
      - step: 5
        name: "Show paraboloid surface"
        description: >
          Draw a translucent surface z = x^2 + y^2.
          Keep data points visible above the surface.
      - step: 6
        name: "Show separating hyperplane"
        description: >
          Draw a translucent horizontal plane z = c, where c is between the blue and red classes.
          Label it as 'Hyperplane in feature space'.
      - step: 7
        name: "Project back to 2D"
        description: >
          Show that z = c and z = x^2 + y^2 imply x^2 + y^2 = c.
          Draw the corresponding decision circle on the original 2D plane.
      - step: 8
        name: "Camera rotation"
        description: >
          Slowly rotate the 3D camera to help students understand the spatial separation.
      - step: 9
        name: "Final summary"
        description: >
          Display:
          'In 3D: linear hyperplane'
          'In 2D: nonlinear decision boundary'
          'This is the intuition behind the kernel trick.'

    manim_requirements:
      scene_class: "SVMKernelTrick3D"
      base_class: "ThreeDScene"
      objects:
        - "ThreeDAxes"
        - "Dot3D"
        - "Surface"
        - "ParametricFunction"
        - "MathTex"
        - "Text"
        - "VGroup"
      visual_style:
        background_color: "dark"
        blue_points_color: "BLUE"
        red_points_color: "RED"
        hyperplane_color: "YELLOW"
        surface_opacity: 0.22
        hyperplane_opacity: 0.35
        decision_boundary_color: "YELLOW"
      camera:
        initial_phi_degrees: 65
        initial_theta_degrees: -45
        ambient_rotation_rate: 0.18

    success_criteria:
      - "The animation clearly shows 2D nonlinearity."
      - "The points visibly lift into 3D according to z = x^2 + y^2."
      - "The 3D hyperplane separates the classes."
      - "The 2D circular decision boundary is shown as projection."
      - "The formulas are readable and synchronized with animation."

  - phase_id: 2
    title: "Real RBF SVM Decision Function Surface"
    objective: >
      Implement a true sklearn SVC model using the RBF kernel and visualize the
      decision function f(x, y) as a 3D surface. The surface should show how the
      RBF kernel creates a nonlinear decision boundary in the original 2D space.

    output_file: "phase2_rbf_decision_surface.py"

    educational_message:
      main_idea: >
        The previous 3D mapping is a teaching-friendly visualization.
        A real RBF kernel does not simply map points to 3D. It implicitly maps
        them to a high-dimensional or infinite-dimensional feature space.
        We visualize the model by plotting the decision function surface.
      key_formulae:
        - "K(x, x') = exp(-gamma ||x - x'||^2)"
        - "f(x) = sum_i alpha_i y_i K(x_i, x) + b"
        - "f(x, y) = 0 is the decision boundary"

    data_design:
      use_same_dataset_as_phase1: true
      random_seed: 7
      noise_option:
        enabled: true
        default_noise: 0.08
        purpose: "Make the dataset more realistic."

    model:
      library: "scikit-learn"
      estimator: "SVC"
      default_params:
        kernel: "rbf"
        C: 10
        gamma: 1
      required_methods:
        - "fit"
        - "decision_function"
        - "predict"
      support_vectors:
        show_support_vectors: true
        marker_style: "larger ring marker"

    visualization:
      backend_options:
        primary: "matplotlib 3D"
        optional: "plotly"
      plots:
        - name: "2D dataset with decision boundary"
          description: >
            Plot original data points in 2D.
            Draw contour line where decision_function = 0.
            Also draw margin contours where decision_function = -1 and +1.
        - name: "3D decision function surface"
          description: >
            Plot z = f(x, y), where f is clf.decision_function.
            Use surface height to represent model confidence.
            Mark z = 0 as the decision surface threshold.
        - name: "support vectors"
          description: >
            Highlight support vectors in both 2D and 3D visualizations.

    implementation_details:
      functions_to_create:
        - name: "generate_ring_dataset"
          location: "utils/data_generator.py"
          input:
            - "n_inner"
            - "n_outer"
            - "inner_radius_range"
            - "outer_radius_range"
            - "noise"
            - "random_seed"
          output:
            - "X"
            - "y"
        - name: "train_svm"
          location: "utils/svm_utils.py"
          input:
            - "X"
            - "y"
            - "kernel"
            - "C"
            - "gamma"
          output:
            - "trained sklearn SVC model"
        - name: "make_decision_grid"
          location: "utils/svm_utils.py"
          input:
            - "x_range"
            - "y_range"
            - "resolution"
          output:
            - "xx"
            - "yy"
            - "grid_points"
        - name: "compute_decision_surface"
          location: "utils/svm_utils.py"
          input:
            - "model"
            - "grid_points"
          output:
            - "Z decision scores reshaped to grid"

    expected_behavior:
      - "When gamma is small, the decision boundary should be smoother."
      - "When gamma is large, the decision boundary should become more flexible and may overfit."
      - "When C is small, margin should be softer and more tolerant."
      - "When C is large, the model should try harder to classify every point correctly."

    success_criteria:
      - "The script trains an actual sklearn SVC with RBF kernel."
      - "The 2D decision boundary is visible."
      - "The 3D decision function surface is visible."
      - "Support vectors are highlighted."
      - "The code clearly distinguishes between educational 3D mapping and real RBF SVM."

  - phase_id: 3
    title: "Interactive Streamlit and Plotly Demo"
    objective: >
      Build an interactive web app where students can adjust SVM parameters
      and immediately see how the decision boundary and 3D decision surface change.

    output_file: "phase3_streamlit_app.py"

    educational_message:
      main_idea: >
        Students should be able to experiment with SVM kernels, C, gamma,
        and dataset noise. The app should help them connect parameter choices
        with decision boundary complexity, margin behavior, and overfitting.

    app_layout:
      title: "Interactive SVM Kernel Trick 3D Demo"
      sidebar_controls:
        - name: "kernel"
          type: "selectbox"
          options:
            - "linear"
            - "poly"
            - "rbf"
            - "sigmoid"
          default: "rbf"
        - name: "C"
          type: "slider"
          min: 0.1
          max: 100.0
          default: 10.0
          step: 0.1
          scale: "log recommended if possible"
        - name: "gamma"
          type: "slider"
          min: 0.01
          max: 10.0
          default: 1.0
          step: 0.01
          visible_when:
            - "kernel == rbf"
            - "kernel == poly"
            - "kernel == sigmoid"
        - name: "degree"
          type: "slider"
          min: 2
          max: 6
          default: 3
          step: 1
          visible_when:
            - "kernel == poly"
        - name: "noise"
          type: "slider"
          min: 0.0
          max: 0.5
          default: 0.08
          step: 0.01
        - name: "number_of_points"
          type: "slider"
          min: 40
          max: 300
          default: 120
          step: 10
        - name: "random_seed"
          type: "number_input"
          default: 7

    main_sections:
      - section_id: "concept_panel"
        title: "Concept"
        content:
          - "2D circular data cannot be separated by a straight line."
          - "Kernel methods allow SVM to learn nonlinear decision boundaries."
          - "RBF kernel uses similarity to support vectors to form a flexible boundary."

      - section_id: "plot_2d"
        title: "2D Decision Boundary"
        description: >
          Display the original points, decision boundary f(x, y) = 0,
          and margin lines f(x, y) = -1 and f(x, y) = +1.
        chart_type: "plotly contour + scatter"

      - section_id: "plot_3d"
        title: "3D Decision Function Surface"
        description: >
          Display z = decision_function(x, y) as an interactive 3D Plotly surface.
          Allow rotation and zoom.
        chart_type: "plotly surface + scatter3d"

      - section_id: "support_vectors"
        title: "Support Vectors"
        description: >
          Show number of support vectors and highlight them on the plots.
        metrics:
          - "number of support vectors"
          - "training accuracy"
          - "kernel"
          - "C"
          - "gamma"

      - section_id: "teaching_notes"
        title: "Teaching Notes"
        content_dynamic_rules:
          - condition: "gamma < 0.2"
            message: "Gamma is small: the boundary is smoother and each point has wider influence."
          - condition: "gamma > 3"
            message: "Gamma is large: the boundary becomes very flexible and may overfit."
          - condition: "C < 1"
            message: "C is small: the model allows more mistakes to keep a wider margin."
          - condition: "C > 20"
            message: "C is large: the model tries harder to classify training data correctly."

    plotly_requirements:
      two_d_plot:
        elements:
          - "blue inner points"
          - "red outer points"
          - "decision boundary contour f = 0"
          - "margin contour f = -1"
          - "margin contour f = +1"
          - "support vectors highlighted"
      three_d_plot:
        elements:
          - "decision function surface z = f(x, y)"
          - "training points placed at z = decision_function(x, y)"
          - "support vectors highlighted"
          - "z = 0 reference plane if possible"

    implementation_details:
      caching:
        use_streamlit_cache_data: true
        use_streamlit_cache_resource: false
      performance:
        grid_resolution_default: 80
        grid_resolution_max: 150
        avoid_extreme_high_resolution: true
      functions_to_reuse:
        - "generate_ring_dataset from utils/data_generator.py"
        - "train_svm from utils/svm_utils.py"
        - "make_decision_grid from utils/svm_utils.py"
        - "compute_decision_surface from utils/svm_utils.py"

    success_criteria:
      - "The Streamlit app runs with one command."
      - "Users can adjust kernel, C, gamma, degree, noise, and number of points."
      - "The 2D decision boundary updates correctly."
      - "The 3D decision function surface updates correctly."
      - "The app clearly explains gamma, C, margin, support vectors, and overfitting."
      - "The interface is suitable for classroom demonstration."

quality_requirements:
  code_quality:
    - "Use clear function names."
    - "Add comments explaining educational purpose."
    - "Avoid overly complex abstractions."
    - "Separate data generation, model training, and visualization."
    - "Keep all scripts executable independently."

  educational_quality:
    - "Do not incorrectly claim that RBF kernel maps data directly to 3D."
    - "Clearly distinguish concept mapping z = x^2 + y^2 from true RBF SVM."
    - "Use formulas and visual explanation together."
    - "Make the demo understandable for students who know basic x-y graphs."

  visual_quality:
    - "Use blue for inner class."
    - "Use red for outer class."
    - "Use yellow for decision boundary and hyperplane."
    - "Use translucent surfaces when needed."
    - "Keep formulas large and readable in Manim."

  testing:
    manual_tests:
      - "Run Manim low quality preview successfully."
      - "Run Manim high quality render successfully."
      - "Run phase2 script and verify 2D and 3D plots."
      - "Run Streamlit app and test all sliders."
      - "Confirm no runtime errors when switching kernels."
      - "Confirm support vectors display correctly."
      - "Confirm decision boundary changes when gamma and C change."

agent_instructions:
  implementation_order:
    - "Create repository structure."
    - "Create requirements.txt."
    - "Implement utils/data_generator.py."
    - "Implement utils/svm_utils.py."
    - "Implement Phase 1 Manim animation."
    - "Implement Phase 2 sklearn RBF decision surface script."
    - "Implement Phase 3 Streamlit interactive app."
    - "Write README.md with setup and run commands."
    - "Perform manual tests and fix errors."

  do_not_do:
    - "Do not merge all code into one giant file."
    - "Do not claim RBF is simply a 3D mapping."
    - "Do not make the Streamlit app depend on Manim."
    - "Do not use random colors that confuse class identity."
    - "Do not omit support vectors."

  final_deliverables:
    - "Working Manim animation script."
    - "Working sklearn RBF SVM decision surface script."
    - "Working Streamlit interactive demo."
    - "Reusable utility functions."
    - "README with explanation and commands."
    - "requirements.txt."

readme_requirements:
  sections:
    - "Project Overview"
    - "Educational Story"
    - "Phase 1: Manim Kernel Trick Animation"
    - "Phase 2: Real RBF SVM Decision Surface"
    - "Phase 3: Interactive Streamlit Demo"
    - "Installation"
    - "Run Commands"
    - "Important Mathematical Note"
    - "Teaching Suggestions"

  important_mathematical_note: >
    The mapping z = x^2 + y^2 is used as a visual and educational feature mapping
    to explain why nonlinear data can become linearly separable in a higher-dimensional
    feature space. A real RBF kernel does not explicitly map data to only 3D;
    it corresponds to a high-dimensional or infinite-dimensional feature space.
    Therefore, the RBF decision surface shown in Phase 2 and Phase 3 visualizes
    the decision function f(x, y), not the full feature space itself.