
apt-get update
apt-get install -y python3 python3-pip

cd $WDIR
pip3 install -r requirements.txt

python3 trainer.py

