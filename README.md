# Synchroflow
OpenBCI Cyton -> Wifi Websocket -> ESP32 LED Glasses

## Setup (Single-user mode)
1. Setup OpenBCI cap with Cyton
- Run `python3 python/brainflow/brain_serve_delta.py` from terminal while on 2.4GHz network
2. Install Arduino IDE
- Upload `Arduino/delta_power_wifi.ino` to ESP32 on LED glasses

## Setup (Paired-user mode)
1. Repeat single-user mode setup on separate host IP and port. 
2. Trade glasses so you get eachothers brain data


## Troubleshooting Tips
1. Install OpenBCI GUI
2. Use `python3 python/brainflow/data_stream_test_bands_agg.py` to test if data is being sent correctly
3. Use `python3 python/brainflow/brainswerve_banks.py` and `python3 python/brainflow/brain_veer_delta.py` to test if data is being received correctly when server from step 1 s running (can't run multiple receivers at the same time very well)
