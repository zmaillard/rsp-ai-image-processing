output "droplet_ip" {
  value = digitalocean_droplet.rsp_image_processing.ipv4_address
}

resource "local_file" "inventory_file" {
  content = templatefile("./inventory.template",
    {
      digitalocean_public_ip = [digitalocean_droplet.rsp_image_processing.ipv4_address]
    }
  )
  filename = "./inventory"
}
