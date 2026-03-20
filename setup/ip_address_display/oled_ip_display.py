#!/usr/bin/env python3
# Copyright (c) 2017 Adafruit Industries
# Author: Tony DiCola & James DeVito
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import time
import platform
import subprocess
import os.path
import traceback

import Adafruit_SSD1306
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

def detect_platform():
    """Detect the platform (Jetson or Raspberry Pi) and return I2C bus number"""
    # Try to detect Jetson first
    try:
        import Jetson.GPIO as GPIO
        board_name = GPIO.gpio_pin_data.get_data()[0]
        print(f"Detected Jetson board: {board_name}")
        
        if board_name == "JETSON_NX":
            return 8, "jetson"
        elif board_name == "JETSON_XAVIER":
            return 8, "jetson"
        elif board_name == "JETSON_NANO":
            return 1, "jetson"
        elif board_name == "JETSON_ORIN":
            return 7, "jetson"
        elif board_name == "JETSON_ORIN_NX":
            return 7, "jetson"
        elif board_name == "JETSON_ORIN_NANO":
            return 7, "jetson"
        else:
            return 1, "jetson"  # Default for unknown Jetson boards
    except:
        pass
    
    # Check if it's a Raspberry Pi
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'Raspberry Pi' in cpuinfo:
                print("Detected Raspberry Pi")
                # Raspberry Pi typically uses I2C bus 1
                return 1, "raspberry"
    except:
        pass
    
    # Default to bus 1 if platform cannot be detected
    print("Platform detection failed, defaulting to I2C bus 1")
    return 1, "unknown"

def get_network_interfaces():
    """Get list of available network interfaces"""
    interfaces = []
    try:
        # Get all network interfaces
        result = subprocess.check_output("ls /sys/class/net/", shell=True).decode('utf-8').strip().split()
        
        # Common interface patterns
        ethernet_patterns = ['eth', 'enp', 'ens', 'enx']
        wifi_patterns = ['wlan', 'wlp', 'wls', 'wlx']
        usb_patterns = ['usb']
        
        # Categorize interfaces
        ethernet = None
        wifi = None
        usb = None
        
        for iface in result:
            if iface == 'lo':  # Skip loopback
                continue
                
            iface_lower = iface.lower()
            
            # Check for ethernet
            if not ethernet and any(pattern in iface_lower for pattern in ethernet_patterns):
                ethernet = iface
            # Check for wifi
            elif not wifi and any(pattern in iface_lower for pattern in wifi_patterns):
                wifi = iface
            # Check for usb
            elif not usb and any(pattern in iface_lower for pattern in usb_patterns):
                usb = iface
        
        return ethernet, wifi, usb
    except:
        return None, None, None

def get_ip_address(interface):
    """Get IP address for a network interface"""
    if interface is None:
        return "N/A"
        
    interface_state = get_network_interface_state(interface)
    if interface_state is None:  # No interface found
        return "N/A"
    elif interface_state == 'down':
        return "DOWN"

    try:
        # Use ip command instead of ifconfig for better compatibility
        cmd = f"ip addr show {interface} | grep -Eo 'inet ([0-9]*\.)+[0-9]*' | grep -Eo '([0-9]*\.)+[0-9]*' | grep -v '127.0.0.1' | head -1"
        result = subprocess.check_output(cmd, shell=True).decode('ascii').strip()
        if result:
            return result
        else:
            return "NO IP"
    except subprocess.CalledProcessError:
        return "ERROR"

def get_network_interface_state(interface):
    """Get the state of a network interface"""
    if os.path.isfile(f'/sys/class/net/{interface}/operstate'):
        try:
            return subprocess.check_output(f'cat /sys/class/net/{interface}/operstate', shell=True).decode('ascii')[:-1]
        except:
            return None
    else:
        return None

def main():
    # Detect platform and get I2C bus number
    i2c_busnum, platform_type = detect_platform()
    print(f"Using I2C bus {i2c_busnum} on {platform_type}")
    
    # 128x32 display with hardware I2C:
    disp = Adafruit_SSD1306.SSD1306_128_32(rst=None, i2c_bus=i2c_busnum, gpio=1)

    # Initialize library.
    disp.begin()

    # Clear display.
    disp.clear()
    disp.display()

    # Create blank image for drawing.
    # Make sure to create image with mode '1' for 1-bit color.
    width = disp.width
    height = disp.height
    image = Image.new('1', (width, height))

    # Get drawing object to draw on image.
    draw = ImageDraw.Draw(image)

    # Draw a black filled box to clear the image.
    draw.rectangle((0, 0, width, height), outline=0, fill=0)

    # First define some constants to allow easy resizing of shapes.
    padding = -1
    top = padding
    bottom = height - padding
    # Move left to right keeping track of the current x position for drawing shapes.
    x = 0

    # Load a smaller font for 4-line display on 128x32 OLED
    try:
        # Try different font sizes - 8pt allows 4 lines on 32px height
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 8)
    except:
        try:
            # Alternative: Liberation Mono which is often clearer at small sizes
            font = ImageFont.truetype('/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf', 8)
        except:
            # Fallback to default font if TTF loading fails
            font = ImageFont.load_default()

    # Get network interfaces
    eth_iface, wifi_iface, usb_iface = get_network_interfaces()
    print(f"Detected interfaces - Ethernet: {eth_iface}, WiFi: {wifi_iface}, USB: {usb_iface}")

    count = 0
    while True:
        # Draw a black filled box to clear the image.
        draw.rectangle((0, 0, width, height), outline=0, fill=0)

        # Get IP addresses for the interfaces
        eth_ip = get_ip_address(eth_iface)
        wifi_ip = get_ip_address(wifi_iface)
        
        # Get system stats
        # Memory usage
        cmd = "free -m | awk 'NR==2{printf \"%d/%dMB\", $3,$2}'"
        mem_usage = subprocess.check_output(cmd, shell=True).decode('utf-8')
        
        # Disk usage
        cmd = "df -h / | awk 'NR==2{printf \"%s/%s\", $3,$2}'"
        disk_usage = subprocess.check_output(cmd, shell=True).decode('utf-8')

        # Display information in 4 lines for 128x32 display
        # Line 1: Ethernet IP
        draw.text((x, top), f"Eth: {eth_ip}", font=font, fill=255)
        # Line 2: WiFi IP  
        draw.text((x, top + 8), f"WiFi:{wifi_ip}", font=font, fill=255)
        # Line 3: Memory usage
        draw.text((x, top + 16), f"Mem: {mem_usage}", font=font, fill=255)
        # Line 4: Disk usage
        draw.text((x, top + 24), f"Disk:{disk_usage}", font=font, fill=255)

        # Display image.
        disp.image(image)
        disp.display()
        
        # Exit logic: wait up to 10 seconds for IPs or exit when found
        if (eth_ip in ['N/A', 'DOWN', 'NO IP'] or wifi_ip in ['N/A', 'DOWN', 'NO IP']) and count < 10:
            time.sleep(1)
            count += 1
        else:
            break

if __name__ == "__main__":
    ########################################
    # OLED I2C address check
    ########################################
    try:
        import smbus
        i2c_busnum, _ = detect_platform()
        bus = smbus.SMBus(i2c_busnum)
        i2c_address = 0x3C
        bus.read_byte(i2c_address)
        main()
    except Exception as e:
        # no oled or other error
        err_info = traceback.format_exc()
        print(err_info)