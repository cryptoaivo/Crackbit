
CrackBit - Ultimate Bitcoin Puzzle Solver
CrackBit is the first open-source solution designed to achieve enterprise-grade blockchain scanning speeds. It leverages CUDA for high-performance processing, making it ideal for solving Bitcoin puzzles efficiently.

License: GPL-3.0
Platform: Windows / Linux
CUDA: 12.2
🚀 Quick Start Guide
1. Dependencies Installation
For Windows:
You'll need the following tools installed on your machine:

PowerShell
NVIDIA CUDA
OpenSSL
CMake
Visual Studio Build Tools
Run the following commands in an Admin PowerShell window to install dependencies:

powershell
Copy
Edit
# Install CUDA
winget install -e --id NVIDIA.CUDA

# Install OpenSSL
winget install -e --id OpenSSL.OpenSSL

# Install CMake
winget install -e --id Kitware.CMake

# Install Visual Studio Build Tools
winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended"
For Linux (Ubuntu/Debian):
Run the following commands to install the required dependencies:

bash
Copy
Edit
sudo apt update && sudo apt install -y \
    nvidia-cuda-toolkit \
    nvidia-driver-535 \
    build-essential \
    libssl-dev \
    libboost-all-dev \
    cmake \
    git \
    ocl-icd-opencl-dev
2. Installation
Step 1 - Clone the Repository:
First, clone the CrackBit repository:

bash
Copy
Edit
git clone --depth 1 https://github.com/crackbit/crackbit.git
cd crackbit
Step 2 - Build with CUDA Optimization:
Create the build directory and compile the project:

bash
Copy
Edit
mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release -j$(nproc)
Step 3 - Create Launchers:
For Linux: Create a launcher script:

bash
Copy
Edit
echo '#!/bin/sh
cd "$(dirname "$0")/build"
./CrackBit "$@"' > ../CrackBit.sh
chmod +x ../CrackBit.sh
For Windows: Create a Launch.bat file:

batch
Copy
Edit
@echo off
cd /d "%~dp0build\Release"
CrackBit.exe %*
pause
3. Usage
Basic Scanning:
To start solving a Bitcoin puzzle, use the following command:

bash
Copy
Edit
./CrackBit.sh --range 9A --puzzle 68
Key Features:
Auto-Resume: Progress is saved every 30 seconds in checkpoint.dat.
Multi-GPU: Distribute the load across multiple GPUs using the --gpus option. Example: --gpus 0,1,2.
Real-Time Stats: Get live monitoring with a built-in dashboard.
4. Advanced Configuration
You can customize CrackBit's behavior by editing the settings.txt file:

ini
Copy
Edit
[GPU]
# 0 = Auto-optimize
threads = 0  
# 2-8 character hex prefix
target = 9A1F

[Performance]
# Keys per GPU batch (default: 1e8)
batch_size = 100000000

[Output]
log_level = detailed
results_file = found.txt
Command Reference:
Option	Description
--range XX	Specify a 2-digit hex starting prefix.
--puzzle N	Select puzzle number (68-75).
--benchmark	Run performance tests.
--api 8080	Enable REST API on port 8080.
--telegram	Connect Telegram bot for notifications.
5. Performance Tuning
Optimal Settings by GPU Model:
GPU Model	Threads	Batch Size	Expected Speed
RTX 4090	1024	1e8	42.6B keys/s
RTX 3090	896	7.5e7	31.2B keys/s
RTX 2080 Ti	768	5e7	18.4B keys/s
Auto-Tune for Your Hardware:
CrackBit can automatically optimize settings for your GPU:

bash
Copy
Edit
./CrackBit.sh --optimize
⚠️ Legal & Compliance
By using CrackBit, you agree to the following terms:

Only scan addresses that you legally own.
Ensure that you comply with jurisdictional regulations regarding blockchain analysis.
Properly report taxes as required by law.
Important: You must comply with CFAA/EUCD regulations and not scan unauthorized addresses.

📜 License
CrackBit is licensed under GPL-3.0 with Commons Clause. Commercial use requires written permission. You can view the full text of the license here.

Need Help?
Open an issue on the GitHub repository.
Join our Discord Support Server.
Visit the Documentation for more details.
Check out our Benchmarks to evaluate performance.
Consider donating to support CrackBit's development.
❓ Frequently Asked Questions (FAQ)
Q: Can I use AMD GPUs?
No, CrackBit currently only supports NVIDIA GPUs (RTX 20xx/30xx/40xx) for CUDA acceleration.

Q: How do I pause and resume the process?
Simply press CTRL+C to stop and save progress. When you relaunch CrackBit, it will resume from where it left off.

Q: Why are the speeds slower than expected?
Use the --benchmark command to verify your CUDA and driver versions. Ensure your hardware is optimized for maximum performance.
