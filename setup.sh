#!/bin/sh

sudo modprobe -v dummy numdummies=2
sudo ip a add 166.166.1.1/24 dev dummy0
sudo ip link set dummy0 up

sudo /usr/bin/vmhgfs-fuse .host:/ /home/corentingouil/shares -o subtype=vmhgfs-fuse,allow_other