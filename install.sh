#!/usr/bin/env bash
# install.sh — XRP Primer AI Kit Demo setup script
# Tested on Ubuntu 22.04+ (Rubik Pi 3)
# Run with: bash install.sh

# This script sets up the XRP Primer AI Kit Demo, including system dependencies,
# auto-login configuration, Python environment setup, and model downloads.

set -e         # Exit immediately if a command exits with a non-zero status.
set -u         # Treat unset variables and parameters as an error.
set -o pipefail # The return value of a pipeline is the status of the last command to exit with a non-zero status.

# --- GLOBAL VARIABLES ---
USER_TO_AUTOLOGIN=$(whoami)
PROJECTS_DIR="$HOME/projects"
REPO_NAME="Primer-Software"
REPO_URL="https://github.com/SaintSampo/Primer-Software"
REPO_PATH="$PROJECTS_DIR/$REPO_NAME"
VENV_PATH="$REPO_PATH/venv"
PYTHON_BIN="$VENV_PATH/bin/python"
PYTHON_SCRIPT_PATH="$REPO_PATH/src/primer.py"

# --- COLOR OUTPUT HELPERS ---
info()    { echo -e "\033[1;34m[INFO]\033[0m $*"; }
success() { echo -e "\033[1;32m[SUCCESS]\033[0m $*"; }
warn()    { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; exit 1; }

# Helper function to check the status of the last command
check_status() {
    local exit_code=$?
    local step_name="$1"
    if [ $exit_code -ne 0 ]; then
        error "$step_name failed with exit code $exit_code."
    fi
}

# --- FUNCTIONS ---

# Downloads a file if it doesn't already exist.
download_model() {
    local url="$1"
    local output_path="$2"
    local filename=$(basename "$output_path")

    # Check if file exists and is not zero size (-s)
    if [ -f "$output_path" ] && [ -s "$output_path" ]; then
        warn "$filename already exists and is non-empty, skipping download."
        return 0
    fi

    info "Downloading $filename from $url..."
    # Use -q (quiet) and --show-progress for a cleaner progress bar
    wget -q --show-progress -O "$output_path" "$url"
    check_status "Download of $filename"

    if [ -s "$output_path" ]; then
        success "$filename downloaded successfully."
    else
        error "Failed to download $filename: File is empty or download failed."
    fi
}

# Configures the autostart .desktop file to run the main script on login.
setup_autostart() {
    info "Setting up autostart entry for '$PYTHON_SCRIPT_PATH'..."

    local AUTOSTART_DIR="$HOME/.config/autostart"
    local SCRIPT_FILENAME=$(basename "$PYTHON_SCRIPT_PATH")
    local DESKTOP_FILENAME="${SCRIPT_FILENAME%.*}.desktop"
    local DESKTOP_PATH="${AUTOSTART_DIR}/${DESKTOP_FILENAME}"

    # 1. Validate the script path
    if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
        error "Autostart failed: Python script not found at '$PYTHON_SCRIPT_PATH'. (Ensure clone step succeeded)"
    fi

    # 2. Create the autostart directory
    mkdir -p "$AUTOSTART_DIR"
    check_status "Creating autostart directory"

    # 3. Define the execution command: Use the virtual environment's python.
    # The command uses 'bash -c' to ensure we change directory, activate the venv,
    # and then execute the script, all in one line for the .desktop file.
    local EXEC_COMMAND="bash -c 'cd \"$REPO_PATH\" && source venv/bin/activate && sudo \"$PYTHON_BIN\" \"$PYTHON_SCRIPT_PATH\"'"

    # 4. Write the .desktop file content
    cat << EOF > "$DESKTOP_PATH"
[Desktop Entry]
Type=Application
Exec=$EXEC_COMMAND
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
Name=$SCRIPT_FILENAME Startup
Comment=Automatically executes the Primer AI script on user login from its virtual environment.
Keywords=AI;Primer;
EOF
    check_status "Creating .desktop file"

    # 5. Set permissions
    chmod +x "$DESKTOP_PATH"
    success "Autostart desktop entry created: $DESKTOP_PATH"

    echo "--- Autostart Command ---"
    echo "This command will run: $EXEC_COMMAND"
    echo "-------------------------"
}

# Grants the current user persistent read/write access to SPI devices via udev rules and group membership.
setup_spi_permissions() {
    info "Starting SPI device permissions setup..."

    # --- CONFIGURATION ---
    local SPI_DEVICE="spidev0.0"  # The specific SPI device file name (e.g., spidev0.0)
    local UDEV_RULE_FILE="/etc/udev/rules.d/99-spi-access.rules"
    local SPI_GROUP="spi"
    local CURRENT_USER=$(whoami)
    # ---------------------

    echo "--- Granting $CURRENT_USER access to SPI devices ---"

    # 1. Create the 'spi' group if it doesn't exist
    info "1. Creating group '$SPI_GROUP' (if it doesn't exist)..."
    sudo groupadd -f $SPI_GROUP
    check_status "Creating group '$SPI_GROUP'"

    # 2. Add the current user to the 'spi' group
    info "2. Adding user '$CURRENT_USER' to group '$SPI_GROUP'..."
    # The -a (append) and -G (groups) options add the user to the specified group without removing them from others.
    sudo usermod -a -G $SPI_GROUP $CURRENT_USER
    check_status "Adding user to group '$SPI_GROUP'"

    # 3. Create a persistent udev rule
    info "3. Creating udev rule to set permissions on boot..."
    # The rule sets the ownership group for all spidev devices to 'spi' with read/write access (MODE="0660").
    echo "SUBSYSTEM==\"spidev\", MODE=\"0660\", GROUP=\"$SPI_GROUP\"" | sudo tee $UDEV_RULE_FILE > /dev/null
    check_status "Creating udev rule file"

    # 4. Apply the new udev rules immediately
    info "4. Reloading and triggering udev rules for immediate effect..."
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    check_status "Applying udev rules"

    # 5. Verify the new device permissions
    echo "5. Verifying permissions for $SPI_DEVICE (look for 'rw-rw----' and group 'spi')..."
    # Check if the device exists before listing
    if [ -e "/dev/$SPI_DEVICE" ]; then
        ls -l /dev/$SPI_DEVICE
    else
        warn "/dev/$SPI_DEVICE not found. Permissions will apply when the device is created/enabled."
    fi

    # 6. Final Instructions
    success "SPI Permissions setup is complete."
}

# Grants the current user persistent read/write access to GPIO devices (like /dev/gpiochipX) via udev rules and group membership.
setup_gpio_permissions() {
    info "Starting GPIO device permissions setup for periphery-python..."

    # --- CONFIGURATION ---
    local GPIO_GROUP="gpio"
    local UDEV_RULE_FILE="/etc/udev/rules.d/98-gpio-access.rules"
    local CURRENT_USER=$(whoami)
    # ---------------------

    echo "--- Granting $CURRENT_USER access to GPIO devices (via /dev/gpiochipX) ---"

    # 1. Create the 'gpio' group if it doesn't exist
    info "1. Creating group '$GPIO_GROUP' (if it doesn't exist)..."
    sudo groupadd -f $GPIO_GROUP
    check_status "Creating group '$GPIO_GROUP'"

    # 2. Add the current user to the 'gpio' group
    info "2. Adding user '$CURRENT_USER' to group '$GPIO_GROUP'..."
    sudo usermod -a -G $GPIO_GROUP $CURRENT_USER
    check_status "Adding user to group '$GPIO_GROUP'"

    # 3. Create a persistent udev rule
    info "3. Creating udev rule to set permissions on boot..."
    # The rule sets the ownership group for all gpiochip devices to 'gpio' with read/write access (MODE="0660").
    echo "SUBSYSTEM==\"gpio\", KERNEL==\"gpiochip*\", MODE=\"0660\", GROUP=\"$GPIO_GROUP\"" | sudo tee $UDEV_RULE_FILE > /dev/null
    check_status "Creating udev rule file for GPIO"

    # 4. Apply the new udev rules immediately
    info "4. Reloading and triggering udev rules for immediate effect..."
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    check_status "Applying udev rules for GPIO"

    # 5. Verify the new device permissions
    echo "5. Verifying permissions for /dev/gpiochip0 (look for 'rw-rw----' and group 'gpio')..."
    if [ -e "/dev/gpiochip0" ]; then
        ls -l /dev/gpiochip0
    else
        warn "/dev/gpiochip0 not found. Permissions will apply when the device is created/enabled."
    fi

    success "GPIO Permissions setup is complete."
}

# -----------------------------------------------------------------------------
## 1. Configure System Session Settings (Screen Blanking & Auto-Login)
# -----------------------------------------------------------------------------
info "Disabling screen blanking and locking for current GNOME session..."
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.desktop.screensaver lock-enabled false
success "Screen blanking and locking disabled."

info "Configuring GDM3 for automatic login for user '$USER_TO_AUTOLOGIN'..."
if [ -f /etc/gdm3/custom.conf ]; then
    # Use sed -E for extended regex to handle commented or existing lines robustly
    sudo sed -i.bak -E \
        -e 's/^#?(AutomaticLoginEnable=).*$/AutomaticLoginEnable=true/' \
        -e "s/^#?(AutomaticLogin=).*$/AutomaticLogin=$USER_TO_AUTOLOGIN/" \
        /etc/gdm3/custom.conf
    check_status "Setting GDM auto-login"
    success "Auto-login set for next reboot. (User: $USER_TO_AUTOLOGIN)"
else
    warn "/etc/gdm3/custom.conf not found. Skipping auto-login configuration."
fi

# -----------------------------------------------------------------------------
## 2. System Update and Dependencies
# -----------------------------------------------------------------------------
# System Update is commented out as in the original script.
# info "Updating system packages..."
# sudo apt update && sudo apt upgrade -y

info "Installing required packages (git, curl, wget, build tools, python3.12, audio dev libs, venv)..."
sudo apt install -y git curl wget python3.12 python3.12-venv build-essential python3-dev libportaudio2 portaudio19-dev
check_status "Installing required packages"
success "System dependencies installed."

# -----------------------------------------------------------------------------
## 3. Configure SPI Device Permissions
# -----------------------------------------------------------------------------
setup_spi_permissions

# -----------------------------------------------------------------------------
## 4. Configure GPIO Device Permissions
# -----------------------------------------------------------------------------
setup_gpio_permissions

# -----------------------------------------------------------------------------
## 5. Clone Repository and Setup Python Virtual Environment
# -----------------------------------------------------------------------------
info "Ensuring projects directory structure exists..."
mkdir -p "$PROJECTS_DIR"
check_status "Creating projects directory"

if [ ! -d "$REPO_PATH" ]; then
    info "Cloning repository to $REPO_PATH..."
    # Execute clone in a subshell so 'cd' doesn't affect the main script
    (
        cd "$PROJECTS_DIR" && git clone "$REPO_URL"
    )
    check_status "Cloning repository"
else
    warn "Repository already exists at $REPO_PATH, skipping clone."
fi

# Move into the repository directory for subsequent steps
cd "$REPO_PATH"

if [ ! -d "venv" ]; then
    info "Creating Python 3.12 virtual environment at $VENV_PATH..."
    python3.12 -m venv venv
    check_status "Creating virtual environment"
else
    warn "Virtual environment already exists, skipping creation."
fi

info "Activating virtual environment and installing Python dependencies..."
# This activation is temporary for the rest of the script's execution
source venv/bin/activate

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    check_status "Installing Python dependencies"
    success "Python dependencies installed."
else
    warn "No requirements.txt found — skipping Python dependency installation."
fi

# -----------------------------------------------------------------------------
## 6. Install Ollama and Download TinyLlama Model
# -----------------------------------------------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
    info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    check_status "Installing Ollama"
else
    warn "Ollama already installed, skipping installation."
fi

info "Downloading TinyLlama model via Ollama (initiated in the background)..."
ollama pull tinyllama &
OLLAMA_PID=$!
warn "TinyLlama download started (PID: $OLLAMA_PID). It may take time, but the script will continue."

# -----------------------------------------------------------------------------
## 7. Download Whisper Models (ONNX)
# -----------------------------------------------------------------------------
info "Setting up Whisper ONNX models..."

WHISPER_DIR="lib/whisper" # Relative to $REPO_PATH
mkdir -p "$WHISPER_DIR"

DECODER_URL="https://huggingface.co/onnx-community/whisper-tiny.en/resolve/main/onnx/decoder_model.onnx"
ENCODER_URL="https://huggingface.co/onnx-community/whisper-tiny.en/resolve/main/onnx/encoder_model.onnx"

download_model "$DECODER_URL" "$WHISPER_DIR/decoder_model.onnx"
download_model "$ENCODER_URL" "$WHISPER_DIR/encoder_model.onnx"

# -----------------------------------------------------------------------------
## 8. Configure Autostart on Login
# -----------------------------------------------------------------------------
setup_autostart

# -----------------------------------------------------------------------------
## 9. Final Instructions
# -----------------------------------------------------------------------------
success "Installation complete!"
echo
echo "To manually run the demo:"
echo "1. Change directory to: cd $REPO_PATH"
echo "2. Activate environment: source venv/bin/activate"
echo "3. Run the script (use sudo if GPIO access is required):"
echo "   sudo $PYTHON_BIN src/primer.py"
echo
warn "The system is configured for automatic login and autostart on the next boot."
warn "You may need to log out and back in, or reboot, for changes to take full effect (especially for the new SPI and GPIO permissions)."
