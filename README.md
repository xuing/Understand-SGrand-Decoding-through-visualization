# Understand SGrand Decoding via visualization

A Python project for learning and implementing SGrand (Soft Maximum Likelihood Decoding using GRAND with abandonment) decoding algorithms for error-correcting codes.

## Overview

Provides a simple and easy-to-use framework for quickly combining codes, channels, and decoders through a runner, offering both single-run execution and Monte Carlo simulation.

Most importantly, it includes an intuitive visualization page for SGrand decoding.

This project for Jaist I437E Coding Theory Final Project Report.


## Project Structure

```
Learn_SGrand_Decoding/
├── main.py                     # Entry point for the application
├── runner.py                   # Test runner and execution orchestration
├── channel.py                  # Channel models (e.g., AWGN)
├── codes.py                    # Error-correcting code implementations
├── decoder.py                  # SGrand decoder implementation
├── sgrand_visualization.html   # Visualization output for decoding results
├── traces/                     # Trace files and logs from experiments
│   ├── awgn_monte_carlo.log
│   └── trace_*.json
└── requirements.txt            # Python dependencies
```

## Supported Codes

- **Hamming codes** (e.g., Hamming(8,4))
- **Golay codes** (e.g., Golay(24,12))

## Installation

[1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Learn_SGrand_Decoding
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. visualization:
  
    sgrand_visualization.html


## Features

- Soft-decision decoding using SGrand algorithm
- AWGN channel simulation
- Monte Carlo simulations for performance analysis
- Query tracing and logging with timestamps
- HTML visualization of decoding results
- Support for multiple error-correcting codes

## Output

- **Trace logs**: JSON files in `traces/` directory with format `trace_<code>_queries<N>_<timestamp>.json`
- **Monte Carlo logs**: `awgn_monte_carlo.log` in `traces/` directory
- **Visualization**: Interactive HTML report in `sgrand_visualization.html`
- **Performance metrics**: BER, decoding success rates, and query counts
