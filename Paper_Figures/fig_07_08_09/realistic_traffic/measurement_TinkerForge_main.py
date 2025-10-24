import json
import os
import re
import subprocess
import time
from multiprocessing import Process
import sys

import numpy as np
from Exscript import Account, Host
from Exscript.util.start import start
from Exscript.protocols import SSH2
from tinkerforge.bricklet_energy_monitor import BrickletEnergyMonitor
from tinkerforge.bricklet_temperature_v2 import BrickletTemperatureV2
from tinkerforge.ip_connection import IPConnection

import signal
import time
import sys

class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, signum, frame):
    self.kill_now = True

def handler(signum, frame):
    raise TimeoutError("timeout")

signal.signal(signal.SIGALRM, handler)

INTERFACE = "eno1"
HOST = "localhost"
PORT = 4223
UID = "UoA"
ssh_output = ""

PKTGEN_PATH: str = "/proc/net/pktgen"
STARTS_WITH_KPKTGEND: re.Pattern = re.compile(r"^kpktgend_\d*$")


def write_pktgen(filepath: str, command: str) -> None:
    os.system("echo \""+str(command)+"\" > "+str(os.path.join(PKTGEN_PATH, filepath)))
    #with open(with open(os.path.join(PKTGEN_PATH, filepath), "w") as file:
    #        file.write(command)
    #        file.close(), "w") as file:
    #    file.write(command)
    #    file.close()
        #subprocess.run(["echo", command], stdout=file)


def setup() -> None:
    subprocess.run(["modprobe", "pktgen"])


def cleanup() -> None:
    for file in filter(STARTS_WITH_KPKTGEND.match, os.listdir(PKTGEN_PATH)):
        write_pktgen(file, "rem_device_all")


def generate(mac_dst: str, ip_src: str, ip_dst: str, udp_src: int, udp_dst: int, packetcount: int, packetrate: str,
             interface: str = INTERFACE, core: int = 0, size: int = 1514) -> None:
    if packetcount != 1:
        os.system("sudo tcpreplay --loop=0 --mbps="+str(packetrate)+" -i "+str(interface)+" ./internet_packet_distribution.pcap")


# End Pktgen interface

def measure_V_I_P(em):
    voltage, current, energy, real_power, apparent_power, reactive_power, power_factor, frequency = em.get_energy_data()
    #temperature = t.get_temperature()
    return ([voltage, current, real_power, apparent_power, reactive_power, power_factor, energy, frequency])


def read_temp(job, host, conn):
    global ssh_output
    conn.execute('show environment temperature')
    ssh_output = repr(conn.response)


def read_rpm(job, host, conn):
    global ssh_output
    conn.execute('show environment fan')
    ssh_output = repr(conn.response)


def read_temp_rpm(account, host):
    try:
        conn = SSH2()             # We choose to use SSH2
        conn.connect(host) # Open the SSH connection
        conn.login(account)       # Authenticate on the remote host
        conn.execute('show environment temperature')
        ssh_output = repr(conn.response)
        conn.send('exit\r')  
        conn.close() 
        components = ssh_output.split(r"\r\n")
        temps = {}
        order = ["sfp", "ddr", "sw2", "cpu", "sw1"]
        for x in range(6, 11):
            temps[order[x - 6]] = components[x].split("    ")[1]

        conn = SSH2()             # We choose to use SSH2
        conn.connect(host) # Open the SSH connection
        conn.login(account)       # Authenticate on the remote host
        conn.execute('show environment fan')
        ssh_output = repr(conn.response)
        conn.send('exit\r')  
        conn.close() 
        components = ssh_output.split(r"\r\n")
        rpm = {}
        for x in range(6, 10):
            rpm[components[x].split("               ")[0]] = components[x].split("               ")[1].split(r"\r")[0]
    except:
        temps=[]
        rpm=[]
    return [temps, rpm]


# all time related units are millisecond
def energy_measurement(duration, filename, em, account, host,killer):
    start = time.time()
    results = {"Start_Time": [], "Time": [], "Voltage": [], "Current": [], "Real_Power": [], "Apparent_Power": [],
               "Apparent_Power_Reactive": [], "Power_Factor": [], "Energy": [], "Frequency": [],"Temp_env":[]}
    results["Start_Time"].append(time.time())
    while time.time() < start + (duration / 1000) and not killer.kill_now:
        time.sleep(2)
        try:
            res = measure_V_I_P(em)
            if (len(res) > 0):
                timestamp = time.time()
                results["Time"].append(timestamp)
                results["Voltage"].append(res[0]/100.0)
                results["Current"].append(res[1]/100.0)
                results["Real_Power"].append(res[2]/100.0)
                results["Apparent_Power"].append(res[3]/100.0)
                results["Apparent_Power_Reactive"].append(res[4]/100.0)
                results["Power_Factor"].append(res[5]/1000.0)
                results["Energy"].append(res[6]/100.0)
                results["Frequency"].append(res[7]/100.0)
        finally:
            with open(str(filename)+ ".json", "w") as jsonfile:
                jsonfile.write(json.dumps(results))
            if killer.kill_now:
                sys.exit(0)
    return results
def device_measurement(duration, filename, account, host,killer):
    start = time.time()
    results = {"Start_Time": [], "Time": [], "Temp": [], "RPM": []}
    results["Start_Time"].append(time.time())
    while time.time() < start + (duration / 1000) and not killer.kill_now:
        try:
            device_metrics = read_temp_rpm(account, host)
            if (len(device_metrics) > 0):
                timestamp = time.time()
                results["Time"].append(timestamp)
                results["Temp"].append(device_metrics[0])
                results["RPM"].append(device_metrics[1])
        finally:
            filename.write(json.dumps(results))
            if killer.kill_now:
                sys.exit(0)
    return results


def runtest(length, mac_dst, ip_src, ip_dst, udp_src, udp_dst, rate, pkt_size, interface, filename, em,account,host,killer,packetcount: int = 0):
    setup()
    generator_proc = Process(target=generate, args=(
        mac_dst, ip_src, ip_dst, udp_src, udp_dst, packetcount, str(rate) + 'M', interface, 0, pkt_size))
    with open(str(filename)+ "dev_metrics.json", "w") as jsonfile:
        ssh_proc=Process(target=device_measurement,args=(length, jsonfile,account,host,killer))
        generator_proc.start()
        ssh_proc.start()
        em.reset_energy()
        energy_measurement(length, filename, em,account,host,killer)
        generator_proc.terminate()
        cleanup()
        ssh_proc.join()



def run_reload(job, host, conn):
    conn.send('reload\r')
    time.sleep(1)
    conn.send('y\r')
    print("rebooting")


if __name__ == '__main__':
    killer = GracefulKiller()
    # For idle test: rate: 1, pkt_size: 64, packetcount: 1
    account = Account('greenfield', '123456')
    host = '10.0.0.1'
    conn = SSH2()             # We choose to use SSH2
    conn.connect(host) # Open the SSH connection
    conn.login(account)       # Authenticate on the remote host
    conn.execute('show version')
    ssh_output = repr(conn.response)
    conn.send('exit\r')  
    print(ssh_output)
    conn.close() 
    setup()
    cleanup()
    ipcon = IPConnection()
    em = BrickletEnergyMonitor(UID, ipcon)
    ipcon.connect(HOST, PORT)
    
    scenario_duration=(90)*1000
    #runtest(scenario_duration, "d8:3a:dd:38:a9:22", "10.0.0.2", "10.0.0.1", "5000", "5000", 1000, 128, INTERFACE,
    #            res_name+"/TEMP", em,account,host,killer)
    
    scenarios=[]
    for br in range(0,1001,80):
            scenarios.append([1,br,64])
    scenarios.append([1,1000,64])
    print(sys.argv[1:])
    res_name=str(int(sys.argv[2]))
    for currun in range(5):
        ind =int(sys.argv[1])
        sc=scenarios[ind]
        print(sc)
        print((ind/len(scenarios))*100)
        #start(account, host, run_reload)
        if sc[1]==0:
            runtest(scenario_duration, "d8:3a:dd:38:a9:22", "10.0.0.2", "10.0.0.1", "5000", "5000", 1, 64, INTERFACE,
            res_name+"/results_"+str(ind)+"_"+str(currun), em,account,host,killer,1)
        else:
            runtest(scenario_duration, "d8:3a:dd:38:a9:22", "10.0.0.2", "10.0.0.1", "5000", "5000", sc[1], sc[2], INTERFACE,
                    res_name+"/results_"+str(ind)+"_"+str(currun), em,account,host,killer)
        time.sleep(10)
