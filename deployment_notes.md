# Deployment Notes & Optimizations

### 1. Requirements Optimization
- Switched `tensorflow` to `tensorflow-cpu==2.10.1`. Streamlit Community Cloud typically runs on CPU-only machines with limited memory (approx 1GB). Using the CPU version reduces the installation size and memory overhead significantly.
- Removed `opencv-python` as it was not used in the codebase (PIL and Numpy handle the image preprocessing), preventing potential `libGL.so.1` system dependency errors common in cloud Linux environments.
- Locked `streamlit==1.24.0` for consistency with the local environment.

### 2. Startup Time Reduction
- Deferred the imports of `reportlab` inside the `generate_pdf_report` function. By not importing PDF generation libraries globally at the top of the file, the initial startup and page reload times are noticeably faster for users until they actually request a PDF export.
- Ensured `@st.cache_resource` is securely set on the `load_model()` function, preventing the Heavy DenseNet121 model from reloading during user interactions (button clicks, image uploads).

### 3. Repository Configuration
- Included a robust `.gitignore` to prevent virtual environments (`venv310/`), cache (`__pycache__/`), and system files from polluting the GitHub repository.
- Intentionally omitted `densenet_best.h5` from `.gitignore` since it is required for Streamlit Cloud deployment (as long as it is <100MB, it can be tracked natively by Git without LFS. It is ~35MB).

### 4. Compatibility Verifications
- Verified that `app.py` properly references `densenet_best.h5` as a local, relative path, which translates flawlessly to a cloud environment workspace.
- The `tf.keras.mixed_precision.Policy('mixed_float16')` policy remains in place to keep the prediction computationally lightweight on the server.
