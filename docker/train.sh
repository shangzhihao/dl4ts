
apt-get update
apt-get install -y python3-venv
python3 -m venv venv
. venv/bin/activate

cd $work_dir
pip install -r requirements.txt
python3 trainer.py

