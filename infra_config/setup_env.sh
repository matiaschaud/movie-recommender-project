#!/bin/bash

# This script automates the setup of a Python virtual environment for Feast.
# It handles virtual environment creation, activation, package installation,
# and Jupyter kernel registration.

# --- Parameters ---
# PROJECT_PATH: The absolute or relative path to your project directory.
#               Example: /home/user/my_feast_project or ../my_feast_project
PROJECT_PATH=${1:-"$(pwd)"} # Default to current directory if not provided

# VENV_NAME: The name of the virtual environment directory.
#            Example: venv
VENV_NAME=${2:-"venv"} # Default to 'venv' if not provided

# REQUIREMENTS_FILE: (Optional) The path to a requirements.txt file.
#                    If provided, packages listed in this file will also be installed.
#                    Example: requirements.txt
REQUIREMENTS_FILE=${3:-""} # Default to empty if not provided
KERNEL_NAME=${4:-$VENV_NAME} # Default to VENV_NAME if not provided

# --- Script Start ---

echo "--- Environment Setup Script ---"
echo "Project Path: ${PROJECT_PATH}"
echo "Virtual Environment Name: ${VENV_NAME}"

# 1. Navigate to your project directory
echo ""
echo "1. Navigating to project directory: ${PROJECT_PATH}"
cd "${PROJECT_PATH}" || { echo "Error: Could not change to directory ${PROJECT_PATH}. Exiting."; exit 1; }
echo "Current directory: $(pwd)"

# 2. Create a Virtual Environment
echo ""
echo "2. Creating virtual environment: ${VENV_NAME}"
if [ -d "${VENV_NAME}" ]; then
    echo "Warning: Virtual environment '${VENV_NAME}' already exists. Skipping creation."
else
    python3 -m venv "${VENV_NAME}" || python -m venv "${VENV_NAME}" || { echo "Error: Failed to create virtual environment. Ensure Python is installed and accessible."; exit 1; }
    echo "Virtual environment '${VENV_NAME}' created successfully."
fi

# 3. Activate the Virtual Environment
echo ""
echo "3. Activating the virtual environment..."
# Check if the activate script exists for Linux/macOS
if [ -f "${VENV_NAME}/bin/activate" ]; then
    source "${VENV_NAME}/bin/activate"
    echo "Virtual environment activated."
# Check for Windows activation (though this script is for bash, it's good to note)
elif [ -f "${VENV_NAME}/Scripts/activate" ]; then
    echo "Note: This script is designed for Linux/macOS. For Windows, please activate manually:"
    echo "  ${VENV_NAME}\\Scripts\\activate"
    # We can't activate Windows style directly in bash, so we'll proceed assuming it's a Linux/macOS system or warn.
    # For cross-platform, one might use a tool like 'conda' or separate scripts.
    # For this bash script, we'll assume the /bin/activate path for activation.
    # If on Git Bash/WSL on Windows, '/bin/activate' path might still work.
    source "${VENV_NAME}/bin/activate" 2>/dev/null # Try activating anyway in case it's WSL/Git Bash
    if [ $? -ne 0 ]; then
      echo "Failed to activate environment. Please activate it manually and re-run the installation steps."
      # Exit or continue with a warning based on desired strictness
      # For now, let's exit as pip install will likely fail without activation.
      exit 1
    fi
else
    echo "Error: Could not find activation script for '${VENV_NAME}'. Exiting."
    exit 1
fi

# 4. Install ipykernel
echo ""
echo "4. Installing ipykernel..."
INSTALL_COMMAND="pip install ipykernel"

if [ -n "${REQUIREMENTS_FILE}" ]; then
    if [ -f "${REQUIREMENTS_FILE}" ]; then
        echo "Installing packages from requirements file: ${REQUIREMENTS_FILE}"
        INSTALL_COMMAND="${INSTALL_COMMAND} -r ${REQUIREMENTS_FILE}"
    else
        echo "Warning: requirements file '${REQUIREMENTS_FILE}' not found. Skipping its installation."
    fi
fi

# Execute the installation command
${INSTALL_COMMAND}
if [ $? -ne 0 ]; then
    echo "Error: Failed to install Python packages. Exiting."
    deactivate 2>/dev/null # Deactivate if possible
    exit 1
fi
echo "ipykernel installed successfully."

# 5. Register the Kernel with Jupyter
echo ""
echo "5. Registering the kernel with Jupyter..."
python -m ipykernel install --user --name="${VENV_NAME}" --display-name="${KERNEL_NAME}"
if [ $? -ne 0 ]; then
    echo "Error: Failed to register Jupyter kernel. Exiting."
    deactivate 2>/dev/null # Deactivate if possible
    exit 1
fi
echo "Jupyter kernel '${KERNEL_NAME}' registered successfully."

echo ""
echo "--- Setup Complete! ---"
echo "You can now launch JupyterLab and select '${KERNEL_NAME}' as your kernel."
echo "To deactivate the environment manually, run: deactivate"

# Keep the environment active if the script finishes successfully,
# or you can add 'deactivate' here if you want it to exit the venv immediately.
# For typical use, you might want to remain in the activated environment after running.

