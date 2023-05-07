#!/bin/bash

cargo build

if [ $? -ne 0 ]; then
    exit
fi

sudo ./target/debug/net_interface dummy0
