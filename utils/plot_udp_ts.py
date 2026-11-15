import matplotlib.pyplot as plt
from datetime import datetime
import re

raw_data = """
13:20:01.849795 IP 192.0.2.1.64033 > 255.255.255.255.5005: UDP, length 8
13:20:01.853521 IP 192.0.2.1.64033 > 255.255.255.255.5005: UDP, length 8
13:20:01.856925 IP 192.0.2.1.64033 > 255.255.255.255.5005: UDP, length 8
13:20:01.860404 IP 192.0.2.1.64033 > 255.255.255.255.5005: UDP, length 8
13:20:01.863782 IP 192.0.2.1.64033 > 255.255.255.255.5005: UDP, length 8
13:20:01.867505 IP 192.0.2.1.64033 > 255.255.255.255.5005: UDP, length 8
13:20:01.870759 IP 192.0.2.1.64033 > 255.255.255.255.5005: UDP, length 8
13:20:01.874505 IP 192.0.2.1.64033 > 255.255.255.255.5005: UDP, length 8
13:20:01.877840 IP 192.0.2.1.64033 > 255.255.255.255.5005: UDP, length 8
13:20:01.881506 IP 192.0.2.1.64033 > 255.255.255.255.5005: UDP, length 8
13:20:01.884949 IP 192.0.2.1.64033 > 255.255.255.255.5005: UDP, length 8
"""

# Extract all timestamps using regex
timestamps_str = re.findall(r"(\d{2}:\d{2}:\d{2}\.\d+)", raw_data)

# Convert to datetime objects for accurate plotting
timestamps = [datetime.strptime(ts, "%H:%M:%S.%f") for ts in timestamps_str]
indices = range(1, len(timestamps) + 1)

# Generate the plot
plt.figure(figsize=(10, 5))
plt.scatter(timestamps, indices, s=15, alpha=0.8, color='blue')
plt.xlabel('Timestamp')
plt.ylabel('Packet Index')
plt.title('Packet Index vs. Timestamp')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()