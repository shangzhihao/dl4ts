
apt-get update
apt-get install -y python3 python3-pip

cd $work_dir
pip3 install -r requirements.txt

python3 trainer.py

