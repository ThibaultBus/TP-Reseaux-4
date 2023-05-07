echo "This script will install the fm_transmitter program on your Raspberry Pi."
echo "It will also install the required dependencies. If you're not running Raspberry Pi OS, you may need to install the dependencies manually."
sudo apt-get update
sudo apt-get install make build-essential git
git clone https://github.com/markondej/fm_transmitter
cd fm_transmitter
make