output "droplet_ip" {
  value = digitalocean_droplet.rsp_image_processing.ipv4_address
}
