#!/bin/bash
sudo apt install -y python3-venv git
python3 -m venv venv
#git clone https://github.com/lsinfo3/switch_spline_power_model.git
source venv/bin/activate
python3 -m pip install -r ./switch_spline_power_model/requirements.txt
cwd=$(pwd)
cd ./switch_spline_power_model/Paper_Figures/
cd fig_02/
python3 plot.py
cp fig_02.pdf $cwd/fig_02.pdf
cd ..

cd fig_05/
python3 plot.py
cp fig_05.pdf $cwd/fig_05.pdf
cd ..

cd fig_06/
python3 plot.py
cp fig_06.pdf $cwd/fig_06.pdf
cd ..

cd fig_07_08_09/
python3 plot.py
cp fig_07.pdf $cwd/fig_07.pdf
cp fig_08.png $cwd/fig_08.png
cp fig_09.pdf $cwd/fig_09.pdf
cd ..

cd fig_10/
python3 plot.py
cp fig_10.pdf $cwd/fig_10.pdf
