import json

import subprocess, re
from multiprocessing import Process

import time
import termios
import os
import signal
import numpy as np

from Exscript import Account, Host
from Exscript.util.start import start

INTERFACE = "eno1"
COM_PORT = "/dev/serial/by-id/usb-GWInstek_GPM-8213_Virtual_ComPort_GEY842876-if00"

def handler(signum, frame):
    print("timeout")
    raise Exception("timeout")

signal.signal(signal.SIGALRM, handler)

def ser_write(command):
    f = os.open(COM_PORT,os.O_RDWR)
    a = termios.tcgetattr(f)
    a[0] &= ~termios.INLCR
    a[0] &= ~termios.ICRNL
    a[3] &= ~termios.ECHO
    try:
        termios.tcsetattr(f, termios.TCSANOW, a)
        s = "*CLS" + '\n'
        os.write(f, s.encode('ascii'))
        s = command + '\n'
        os.write(f, s.encode('ascii'))
    finally:
        os.close(f)



def ser_read(length=4000):
    f = os.open(COM_PORT,os.O_RDWR)
    a = termios.tcgetattr(f)
    a[0] &= ~termios.INLCR
    a[0] &= ~termios.ICRNL
    a[3] &= ~termios.ECHO
    try:
        termios.tcsetattr(f, termios.TCSANOW, a)
        r = os.read(f, length)
        toreturn = r.decode('ascii')
    except:
        toreturn=""
    finally:
        os.close(f)
        return toreturn

def reset_device(voltage_range,current_range):
    signal.alarm(0)
    # requires root
    os.system("find /sys/bus/usb/devices/*/authorized -exec sh -c 'echo 0 > ${0}; echo 1 > ${0}' {} \;")
    time.sleep(10)
    ser_write("*RST")
    time.sleep(10)
    ser_write(":INPUT:VOLTAGE:RANGE "+str(voltage_range))
    ser_write(":INPUT:CURRENT:RANGE "+str(current_range)+"mA")
    time.sleep(1)


def measure_V_I_P(com_port):
    signal.alarm(2)
    try:
        ser_write(":NUMERIC:NORMAL:PRESET 4")
        ser_write(":NUMERIC:NORMAL:VALUE?")
        raw_data = str(ser_read())
        data_string = raw_data.split(",")
        v = float(data_string[0])  # V
        i = float(data_string[1])  # A
        p = float(data_string[2])  # W
        s = float(data_string[3])  # VA
        q = float(data_string[4])  # VAR
        power_factor = float(data_string[5])  # PF
        thd_u = float(data_string[24])  # thdv
        thd_i = float(data_string[25])  # thdi
        # pmpeak=float(data_string[14].split("\\")[0]) #pmpeak
        return ([v, i, p, s, q, power_factor, thd_u, thd_i])
    except Exception as exc:
        print(exc)
    signal.alarm(0)
    return([])

PKTGEN_PATH: str = "/proc/net/pktgen"
STARTS_WITH_KPKTGEND: re.Pattern = re.compile(r"^kpktgend_\d*$")


def write_pktgen(filepath: str, command: str) -> None:
    with open(os.path.join(PKTGEN_PATH, filepath), "w") as file:
        subprocess.run(["echo", command], stdout=file)


def setup() -> None:
    subprocess.run(["modprobe", "pktgen"])


def cleanup() -> None:
    for file in filter(STARTS_WITH_KPKTGEND.match, os.listdir(PKTGEN_PATH)):
        write_pktgen(file, "rem_device_all")


def generate(mac_dst: str, ip_src: str, ip_dst: str, udp_src: int, udp_dst: int, packetcount: int, packetrate: str,
             interface: str = INTERFACE, core: int = 0, size: int = 1514) -> None:
    print("waiting for reboot")
    time.sleep(120)
    print("start generating packets")
    write_pktgen(f"kpktgend_{core}", f"add_device {interface}@{core}")
    write_pktgen(f"{interface}@{core}", f"dst_mac {mac_dst}")
    write_pktgen(f"{interface}@{core}", f"dst_min {ip_dst}")
    write_pktgen(f"{interface}@{core}", f"dst_max {ip_dst}")
    write_pktgen(f"{interface}@{core}", f"src_min {ip_src}")
    write_pktgen(f"{interface}@{core}", f"src_max {ip_src}")
    write_pktgen(f"{interface}@{core}", f"udp_src_min {udp_src}")
    write_pktgen(f"{interface}@{core}", f"udp_src_max {udp_src}")
    write_pktgen(f"{interface}@{core}", f"udp_dst_min {udp_dst}")
    write_pktgen(f"{interface}@{core}", f"udp_dst_max {udp_dst}")
    write_pktgen(f"{interface}@{core}", f"pkt_size {size}")
    write_pktgen(f"{interface}@{core}", f"rate {packetrate}")
    write_pktgen(f"{interface}@{core}", f"count {packetcount}")
    write_pktgen("pgctrl", "start")


# End Pktgen interface

# all time related units are millisecond
def energy_measurement(duration,filename):
    start=time.time()
    results = {"Start_Time":[],"Time":[],"Voltage": [], "Current": [], "Real_Power": [],"Apparent_Power":[], "Apparent_Power_Reactive":[],"Power_Factor":[],"THD_V":[],"THD_I":[]}
    results["Start_Time"].append(time.time())
    while time.time()<start+(duration/1000):
        res=measure_V_I_P(COM_PORT)
        if(len(res)>0):
            timestamp=time.time()
            results["Time"].append(timestamp)
            results["Voltage"].append(res[0])
            results["Current"].append(res[1])
            results["Real_Power"].append(res[2])
            results["Apparent_Power"].append(res[3])
            results["Apparent_Power_Reactive"].append(res[4])
            results["Power_Factor"].append(res[5])
            results["THD_V"].append(res[6])
            results["THD_I"].append(res[7])
    with open(filename, "w") as jsonfile:
        jsonfile.write(json.dumps(results))
    return results


def runtest(length, mac_dst, ip_src, ip_dst, udp_src, udp_dst, rate,pkt_size, interface, filename,packetcount:int = 0):
    setup()
    generator_proc = Process(target=generate,
                             args=(
                             mac_dst, ip_src, ip_dst, udp_src, udp_dst, packetcount, str(rate) + 'M', interface, 0,
                             pkt_size))
    generator_proc.start()
    energy_measurement(length,filename)
    generator_proc.terminate()
    cleanup()

def run_reload(job, host, conn):
    conn.send('reload\r')
    time.sleep(1)
    conn.send('y\r')
    print("rebooting")


if __name__ == '__main__':
    account = Account('admin', 'admin')
    host = Host('telnet://10.0.0.3')
    host.set_account(account)
    setup()
    cleanup()
    configs=[[976562.5, 125000000, 128], [977391.304347826, 562000000, 575], [977517.1065493646, 1000000000, 1023]]
    for run in range(5):
        for curr in range(len(configs)):
            reset_device(300,50)
            start(account, host, run_reload)
            runtest(300000, "d8:3a:dd:38:a9:22", "10.0.0.2", "10.0.0.1", "5000", "5000",1,64, INTERFACE,
                     "results_idle_"+str(run)+".json",1)
            # Packet size  half of max_size
            runtest(300000, "d8:3a:dd:38:a9:22", "10.0.0.2", "10.0.0.1", "5000", "5000",configs[curr][1]/1000000,configs[curr][2], INTERFACE,
                        "results_" + str(curr) + "_"+str(run)+".json")
