git init
git config --global user.name swapniljyt
git config --global user.email swapniljytkd888@gmail.com
conda init 
conda create -n mchatbot python=3.8 -y
source activate mchatbot
pip install -r requirements.txt
python template.py