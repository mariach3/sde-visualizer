# 📈 SDE Visualizer — Black–Scholes & Heston

An interactive **Streamlit** app to explore stochastic differential equations (SDEs) in finance.  
The tool demonstrates the **Black–Scholes (GBM)** and **Heston** models through simulations, performance benchmarks, and volatility smile analysis using real AAPL option data.  

This project was developed as part of my MSc thesis *"Stochastic Differential Equations and their Applications in Finance"*.

---

## 🚀 Features

- **SDE Visualiser**
  - Simulate asset price paths under GBM and Heston
  - Compare terminal return distributions vs. Normal
  - Adjustable parameters: drift, volatility, correlation, mean reversion, etc.

- **Performance & Benchmark**
  - Run timing tests for GBM vs Heston simulations
  - Pre-computed scaling plots to understand runtime growth

- **Volatility Smile Explorer**
  - Load real AAPL option data
  - Compare market implied volatility against Black–Scholes and Heston model fits
  - Inspect calibration diagnostics (residual plots)

---

## 🛠 Installation & Running Locally

Clone the repository:

```bash
git clone https://github.com/mariach3/sde-visualizer.git
cd sde-visualizer

