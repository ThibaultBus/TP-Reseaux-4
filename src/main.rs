pub mod network;
pub mod simple_fm;
use fm_data_transmitter::send_file;
use network::process_packet;

extern crate pnet;

use pnet::datalink::{self, NetworkInterface};

use std::env;
use std::io::{self, Read, Write};
use std::process;

use crate::network::save_packet;

fn main() {
    use pnet::datalink::Channel::Ethernet;

    let iface_name = match env::args().nth(1) {
        Some(n) => n,
        None => {
            writeln!(io::stderr(), "USAGE: packetdump <NETWORK INTERFACE>").unwrap();
            process::exit(1);
        }
    };
    let interface_names_match = |iface: &NetworkInterface| iface.name == iface_name;

    // Find the network interface with the provided name
    let interfaces = datalink::interfaces();
    let interface = interfaces
        .into_iter()
        .filter(interface_names_match)
        .next()
        .unwrap_or_else(|| panic!("No such network interface: {}", iface_name));

    // Create a channel to receive on
    let (_, mut rx) = match datalink::channel(&interface, Default::default()) {
        Ok(Ethernet(tx, rx)) => (tx, rx),
        Ok(_) => panic!("packetdump: unhandled channel type"),
        Err(e) => panic!("packetdump: unable to create channel: {}", e),
    };

    let interface2 = interface.clone();

    // FM Receive thread
    std::thread::spawn(|| {
        simple_fm::main();
    });

    // Process received packets thread
    std::thread::spawn(move || loop {
        let rd = std::fs::read_dir("/tmp/packets/recv").unwrap();

        for dir_entry in rd.into_iter() {
            let dir_entry = match dir_entry {
                Ok(dir_entry) => dir_entry,
                Err(_) => continue,
            };

            let mut file = match std::fs::File::open(dir_entry.path()) {
                Ok(file) => file,
                Err(_) => continue,
            };
            let mut buf = [0_u8; 1500];

            loop {
                let read = file.read(&mut buf).unwrap();
                if read == 0 {
                    break;
                }
            }

            process_packet(&buf, &interface2);

            std::fs::remove_file(dir_entry.path());
        }
    });

    let mut file_index = 0;

    loop {
        match rx.next() {
            Ok(packet) => {
                let filepath = format!("/tmp/packets/send/{file_index}");
                save_packet(packet, &filepath);
                std::thread::spawn(move || send_file(&filepath));
            }
            Err(e) => panic!("packetdump: unable to receive packet: {}", e),
        }
    }
}
