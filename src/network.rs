extern crate pnet;

use pnet::datalink::NetworkInterface;

use pnet::packet::ethernet::{EtherTypes, EthernetPacket};
use pnet::packet::ip::IpNextHeaderProtocols;
use pnet::packet::ipv4::Ipv4Packet;
use pnet::packet::{Packet, PacketSize};
use pnet::transport::{transport_channel, TransportChannelType};

use std::io::Write;
use std::net::IpAddr;

fn handle_ipv4_packet(interface_name: &str, ethernet: &EthernetPacket) {
    let header = Ipv4Packet::new(ethernet.payload());
    if let Some(header) = header {
        let (mut tx, _) = match transport_channel(
            header.packet_size(),
            TransportChannelType::Layer3(IpNextHeaderProtocols::Udp),
        ) {
            Ok((tx, rx)) => (tx, rx),
            Err(_) => todo!(),
        };

        let ip_packet = Ipv4Packet::new(header.packet()).unwrap();
        let dest: IpAddr = IpAddr::V4(ip_packet.get_destination());

        tx.send_to(ip_packet, dest);
    } else {
        println!("[{}]: Malformed IPv4 Packet", interface_name);
    }
}

fn handle_ipv6_packet(interface_name: &str, ethernet: &EthernetPacket) {
    todo!()
}

fn handle_arp_packet(interface_name: &str, ethernet: &EthernetPacket) {
    todo!()
}

pub fn process_packet(packet: &[u8], interface: &NetworkInterface) {
    let ethernet_packet = EthernetPacket::new(packet).unwrap();
    let interface_name = &interface.name[..];

    match ethernet_packet.get_ethertype() {
        EtherTypes::Ipv4 => handle_ipv4_packet(interface_name, &ethernet_packet),
        EtherTypes::Ipv6 => handle_ipv6_packet(interface_name, &ethernet_packet),
        EtherTypes::Arp => handle_arp_packet(interface_name, &ethernet_packet),
        _ => println!(
            "[{}]: Unknown packet: {} > {}; ethertype: {:?} length: {}",
            interface_name,
            ethernet_packet.get_source(),
            ethernet_packet.get_destination(),
            ethernet_packet.get_ethertype(),
            ethernet_packet.packet().len()
        ),
    }
}

pub fn save_packet(packet: &[u8], filepath: &str) {
    let mut file = std::fs::File::create(filepath).unwrap();

    let bytes = file.write(packet).unwrap();
    file.flush().unwrap();
}
