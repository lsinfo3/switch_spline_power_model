# A General and Comprehensive Power Consumption Model for Ethernet Switches

This repository contains the source code and measurement results used to measure and determine the parameters of the power consumption model defined in the paper "A General and Comprehensive Power Consumption Model for Ethernet Switches". The structure is as follows:

## Paper_Figures

This directory contains all scripts and measurement results related to the corresponding figure in the paper, including intermediate steps and other artifacts (e.g. the spline object for the MS7048 and the pcap file used to simulate realistic traffic). Due to differences in floating point handling, the exact parameters obtained from rerunning the individual scripts may not match the exact parameters in the scripts. We have observed differences between 9.th generation, 12th, and 13.th generation Intel Core processors, as well as different results in Linux and Windows. All results in the paper are obtained running Ubuntu 24.04.3 LTS on a Core i7-9700.

## General_Scripts

This folder contains scripts, which can be used to measure additional switches. Evaluation can then be performed by the scripts in Paper_Figures. We observed issues with closing the files for the NET_PKTGEN traffic generator for long measurement periods, solved by separating the measurement procedure. To start the measurement "run_measurement.py" needs to be run with root privileges on a Linux system.
