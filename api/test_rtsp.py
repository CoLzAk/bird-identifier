#!/usr/bin/env python3
"""
Test RTSP connection
Quick script to test if your RTSP camera is accessible
"""

import os
import cv2
from dotenv import load_dotenv

load_dotenv()

rtsp_url = os.getenv('RTSP_URL')
print(f"Testing RTSP connection to: {rtsp_url[:30]}...")

# Set FFMPEG options
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|timeout;5000000'

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Failed to open RTSP stream")
    print("\nTroubleshooting steps:")
    print("1. Check if the camera is online (ping 192.168.1.74)")
    print("2. Verify the RTSP URL and credentials")
    print("3. Try accessing the stream with VLC or ffplay")
    print("4. Check if the camera supports RTSP on port 554")
    print("5. Ensure no firewall is blocking the connection")
    exit(1)

print("✅ RTSP stream opened successfully")

# Try to read some frames
for i in range(10):
    ret, frame = cap.read()
    if ret:
        print(f"✅ Frame {i+1} read successfully - Shape: {frame.shape}")
    else:
        print(f"❌ Failed to read frame {i+1}")
        break

cap.release()
print("\n✅ Test completed")
