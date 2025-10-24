import random
import numpy as np
import matplotlib.pyplot as plt
from scapy.all import Ether, IP, UDP, wrpcap, Raw

plt.rcParams.update({'font.size': 24})
plt.rcParams['figure.figsize'] = [10, 6]
fig, ax = plt.subplots()
colors = plt.cm.copper(np.linspace(0,1,5))

random.seed(42)
Packet_Length_Ranges = [[64,127],[128,255],[256,511],[512,1023],[1024,1513],[1514,1514],[1515,1518]]
Range_Probabilities=[29.3,6.2,3.2,3.8,36.6,2.0,18.9]
Range_Probabilities_sum=np.cumsum(Range_Probabilities)
print(Range_Probabilities_sum)
packet_sizes=[]
for x in range(10000):
    rand_number=random.random()*100
    chosen_range=0
    for curr in range(len(Range_Probabilities_sum)-1,-1,-1):
        if rand_number<=Range_Probabilities_sum[curr]:
            chosen_range=curr
    print(chosen_range)
    packet_size=random.randint(Packet_Length_Ranges[chosen_range][0],Packet_Length_Ranges[chosen_range][1])
    packet_sizes.append(packet_size)

print(np.median(packet_sizes))
print(np.std(packet_sizes))

plt.ecdf(packet_sizes,color=colors[0])

plt.xlabel('packet size [B]')
plt.ylabel('ECDF')
#ax.set_xticklabels(Ticklabels)
#plt.xscale('log')
plt.grid(which='major')
#plt.legend(loc="lower right",ncol=2,bbox_to_anchor=(1.1,1))
#ax.set_xticks(np.arange(0,1000, 100))
#plt.xticks(rotation=70)
ax.grid(which='major', alpha=0.2)
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
plt.gcf().subplots_adjust(top=0.8)
#plt.show()
plt.savefig('internet_size_distribution.pdf')
plt.close()

def craft_packet(
    src_mac: str,
    dst_mac: str,
    src_ip: str,
    dst_ip: str,
    src_port: int,
    dst_port: int,
    total_size: int,  # Total packet size in bytes
):
    # Build Ethernet / IP / UDP headers
    ether = Ether(src=src_mac, dst=dst_mac)
    ip = IP(src=src_ip, dst=dst_ip)
    udp = UDP(sport=src_port, dport=dst_port)
    
    # Calculate header sizes
    header_size = len(ether / ip / udp)
    
    # Determine payload size to reach desired total packet size
    if total_size < header_size:
        raise ValueError(f"Total size {total_size} is smaller than header size {header_size}")

    payload_size = total_size - header_size - 18
    payload = Raw(load=b"B" * payload_size)

    # Build full packet
    packet = ether / ip / udp / payload

    return packet

# === Packet generation ===

packets=[]
for ps in packet_sizes:
    packets.append(craft_packet(
        src_mac="04:0e:3c:c7:d7:b4",
        dst_mac="d8:3a:dd:38:a9:22",
        src_ip="10.0.0.2",
        dst_ip="10.0.0.1",
        src_port=5000,
        dst_port=5000,
        total_size=ps
    ))
wrpcap("internet_packet_distribution.pcap",packets)
