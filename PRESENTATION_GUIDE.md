# CSE 438: Digital Image Processing — Final Project Guide
## "Image Restoration & Deblurring System"
*A plain-English, step-by-step presentation and setup guide for you and your teammates.*

---

## 0. How Your Teammates Can Clone & Run This on Their Computer

If you push this project to GitHub, your friends can set it up on their own laptops (Windows, Mac, or Linux) in under 3 minutes:

### Step 1: Clone the Repository
```bash
git clone https://github.com/<YOUR-GITHUB-USERNAME>/ImageRestorationSystem.git
cd ImageRestorationSystem
```

### Step 2: Create a Virtual Environment
- **On Windows (PowerShell or CMD):**
  ```powershell
  python -m venv .venv
  ```
- **On Mac / Linux (Terminal):**
  ```bash
  python3 -m venv .venv
  ```

### Step 3: Activate the Virtual Environment
- **On Windows PowerShell:**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(Note: If Windows says "running scripts is disabled", run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in PowerShell, or use Command Prompt: `.\.venv\Scripts\activate.bat`)*
- **On Mac / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Launch the Web App!
```bash
streamlit run app.py
```
*Your browser will automatically open `http://localhost:8501`.*

### Step 6: Run Tests to Verify Everything Works
```bash
pytest tests/ -v
```
*(All 77 unit tests should pass with green checks).*

---

## 1. The Big Picture (What is the End Goal?)

### The 30-Second Elevator Pitch
> *"Cameras and sensors aren't perfect. In the real world, photos get ruined by two things: **noise** (grain, sensor heat, transmission errors) and **blur** (shaky hands, motion, out-of-focus lenses).*  
> *Most people today use AI or neural networks to fix images, but AI is a 'black box' that hallucinates details that weren't there.*  
> *In this project for CSE 438, we built a classical restoration system from scratch using pure mathematics and NumPy. We simulate exact physical degradations (Gaussian noise, impulse noise, lens blur, motion blur) and use classical spatial and Fourier frequency algorithms to mathematically recover the original image, measuring exact PSNR and SSIM quality gains."*

---

## 2. The 5 Core Algorithms Explained in Plain English

| Algorithm | How it Works in Simple Words | Best Used For | Why it Fails / Limitation |
| :--- | :--- | :--- | :--- |
| **Mean Filter** | Looks at a $3\times3$ box around every pixel and replaces the center with the **average** of its neighbors. | General sensor noise. | Blurs sharp edges. If you have black/white noise dots, it smears them into ugly gray spots! |
| **Gaussian Filter** | Like the mean filter, but pixels closer to the center get **more weight** following a bell curve, and pixels further away get less weight. | Fine electronic thermal noise. | Softer and more natural than the mean filter, but still slightly softens sharp boundaries. |
| **Median Filter** | Takes all pixels in the neighborhood, **sorts them from smallest to largest**, and picks the exact **middle number**. | **Salt-and-Pepper (Impulse) Noise.** | Cannot fix lens blur or motion blur. |
| **Inverse Filter** | Takes the blurred image into the frequency domain (Fourier) and **divides** by the blur function: $\hat{F} = G / H$. | Pure optical blur with **zero noise**. | **Catastrophic failure if there is even 1% noise!** Division by near-zero frequencies blows noise up into an unrecognizable blizzard. |
| **Wiener Filter** | The smart version of the inverse filter. Adds a noise-to-signal parameter $K$ to the denominator: $W = \frac{H^*}{\|H\|^2 + K}$. | **Blur + Noise combined.** | Needs a rough guess of the blur kernel and noise level $K$. |

---

## 3. The Two Big Questions Your Faculty Will Ask

### Question 1: "Why does the Median Filter destroy Salt-and-Pepper noise, but the Mean Filter fails?"
- **The Simple Answer:**  
  Salt-and-pepper noise produces extreme values: pure black ($0$) or pure white ($255$).  
  When you use a **Mean Filter**, it adds up those $0$s and $255$s and divides by 9. This spreads the bad pixel into the clean neighboring pixels, creating blurry gray splotches.  
  When you use a **Median Filter**, it sorts the 9 values in order. The $0$s are pushed to the very bottom, and the $255$s are pushed to the very top. As long as less than half the neighborhood is noisy, the middle number is guaranteed to be a clean, uncorrupted pixel! That is why Median filter achieves **$+12.27$ dB PSNR gain** and a near-perfect **$0.98$ SSIM**.

### Question 2: "Why does Direct Inverse Filtering fail in real life?"
- **The Simple Answer:**  
  Blur acts like a low-pass filter: it kills high frequencies. That means at high frequencies, the blur transfer function $H(u, v)$ is almost zero (like $0.00001$).  
  The inverse filter equation is:
  $$\hat{F}(u, v) = F(u, v) + \frac{N(u, v)}{H(u, v)}$$
  If there is even a tiny bit of sensor noise $N = 0.01$, dividing by $0.00001$ multiplies the noise by $1,000\times$! The noise completely swallows the image into a blizzard of static (PSNR drops by $-15.6$ dB).  
  **Wiener filtering fixes this** because its denominator has $+K$. When $H \to 0$, it doesn't divide by zero; it smoothly turns down the gain and suppresses the noise.

---

## 4. What Do the Numbers in "Quantitative Quality Evaluation" Mean?

When you run a restoration in the studio, you see a row of metric cards:
```
Restored PSNR:   28.77 dB  (+13.50 dB)
Restored SSIM:   0.9922    (+0.5714)
CPU Latency:     10.50 ms
```

Here is exactly what to tell your teacher:

### 1. Restored PSNR: 28.77 dB
- **Stands for:** Peak Signal-to-Noise Ratio (measured in decibels, dB).
- **In simple words:** It measures how strong the clean image signal is compared to the corrupting noise.
- **How to read the score:**
  - Below 20 dB: Very noisy / poor quality.
  - 25 to 30 dB: Good, clear restoration.
  - Above 30 dB: Excellent (almost identical to original).
  - **28.77 dB** proves the filter successfully removed almost all noise.

### 2. +13.50 dB (The Green Delta / Improvement)
- **What it means:** This is the **quality boost**: $\Delta\text{PSNR} = \text{PSNR}_{\text{restored}} - \text{PSNR}_{\text{degraded}}$.
- Before filtering, the noisy image had a terrible PSNR of only $15.27\text{ dB}$.
- Your filter increased the signal quality by **$+13.50\text{ dB}$**.
- *Viva tip to impress the professor:* In decibels (a logarithmic scale), every $+3\text{ dB}$ cut the noise power in half. A $+13.5\text{ dB}$ boost means you reduced the noise power by more than **$20\times$**!

### 3. Restored SSIM: 0.9922
- **Stands for:** Structural Similarity Index Measure.
- **In simple words:** While PSNR only looks at raw pixel math, SSIM models how the **human eye** sees edges, contrast, and shapes.
- **Scale:** Ranges from $0.0$ to $1.0$ (where $1.0 = 100\%$ perfect identical match).
- **0.9922** means your restored image has a **99.2% structural match** with the original clean image. Edges and textures were preserved almost perfectly.

### 4. +0.5714 (The Green Delta / Improvement)
- **What it means:** The noisy image had an SSIM of only $0.42$ ($42\%$ match—a noisy mess).
- Your filter boosted human visual similarity by **$+57.1\%$** (from $0.42$ up to $0.992$).

### 5. CPU Latency: 10.50 ms
- **Stands for:** The exact execution time in milliseconds ($10.5\text{ ms} = 0.0105\text{ seconds}$) measured on your CPU.
- **Why faculty care:** In computer engineering, an algorithm is useless if it takes 5 minutes to run. Running in $10.5\text{ ms}$ proves your algorithm is fast enough for **real-time 90+ FPS video processing** and runs efficiently without needing an expensive GPU.

---

## 5. Tomorrow's Demonstration Script (Step-by-Step)

Follow this exact 3-minute sequence when presenting to your teacher:

### Step 1: Open the Application
Run in your PowerShell terminal:
```powershell
.\.venv\Scripts\streamlit run app.py
```
Open `http://localhost:8501`. Show the clean, professional header:  
*"CSE 438: Digital Image Processing — Classical Restoration System"*.

### Step 2: Live Drag & Drop Demo (Tab 1: Interactive Restoration Studio)
1. Select the pre-loaded image `checkerboard.png` or drop any picture you have.
2. Under **Degradation Model**, choose **Salt-and-Pepper Noise (10%)**.
3. Under **Restoration Algorithm**, first select **Mean Filter (3×3)**.  
   *Say:* "Notice how the black and white spikes got smeared into blurry gray smudges, and PSNR is only 17 dB."
4. Now switch the algorithm to **Median Filter (3×3)**.  
   *Say:* "Now watch the Median filter. The spikes completely disappear, the sharp checkerboard corners remain 100% crisp, and PSNR jumps to over 26 dB."

### Step 3: Show the Frequency Deblurring (Inverse vs Wiener)
1. Still in Tab 1, switch the degradation to **Motion Blur (L=11, 45°)** with a little sensor noise ($\sigma = 0.01$).
2. Select **Direct Inverse Filter**.  
   *Say:* "Notice how the image is completely destroyed by noise. This proves the theoretical limitation of unconstrained inverse filtering."
3. Switch to **Wiener Filter ($K=0.01$)**.  
   *Say:* "With the Wiener filter, the $+K$ term prevents division by zero. The motion blur is cleanly removed, edges are recovered, and noise is suppressed."

### Step 4: Show Your Own Restored Images (Tab 2: Compare Restored Images)
1. Click on **Tab 2: Compare Images & Error Heatmap**.
2. Drag and drop your **Clean Reference Image** on the left.
3. Drag and drop your **Restored Image** on the right.
4. Point out the live **PSNR**, **SSIM**, and the **Error Heatmap** that shows exactly how close your restored image is to the original.

### Step 5: Real Historical Photos (Tab 3: Real Historical Photographs)
1. Click on **Tab 3: Real Historical Photos**.
2. Select `cameraman_1930s.png` or `historic_moon_1960s.png`.
3. Point out: *"Sir, because these are genuine historical photographs from the 1930s and 1960s, there is no clean original ground-truth. Following academic integrity rules, we report zero fake metrics and evaluate based on visible contour recovery and grain suppression."*

### Step 6: Proof that Code is Custom (Unit Tests)
If the faculty asks if you wrote the code or used ready-made OpenCV functions:
Open terminal and run:
```powershell
.\.venv\Scripts\pytest tests/ -q
```
Show that **all 77 tests pass** in ~3 seconds, verifying custom 2D convolution, PSF generation, and filter mathematics.

---

## 6. Summary Table for Quick Reference

| Degradation | Best Classical Fix | Metric Improvement | Key Viva Phrase to Remember |
| :--- | :--- | :--- | :--- |
| **Salt & Pepper Noise** | **Median Filter** | **+12.27 dB PSNR**, SSIM: 0.98 | *"Order-statistic filter that rejects extreme outliers."* |
| **Gaussian Sensor Noise** | **Gaussian Filter** | Smooths noise gracefully | *"Decays smoothly with distance; no sinc ringing."* |
| **Motion Blur + Noise** | **Wiener Filter** | **+3.15 dB PSNR**, sharp edges | *"Minimum Mean Square Error; K prevents noise explosion."* |
| **Unpaired Old Photos** | **Selective Filter** | Qualitative visual appraisal | *"No ground truth $\to$ zero fabricated metrics."* |
