resource "digitalocean_droplet" "rsp_image_processing" {
  image    = "gpu-h100x1-base"
  name     = "rsp-image-processing"
  region   = "tor1"
  size     = "gpu-6000adax1-48gb"
  ssh_keys = [data.digitalocean_ssh_key.terraform.id]
}

resource "null_resource" "wait_for_droplet" {
  depends_on = [digitalocean_droplet.rsp_image_processing]

  provisioner "remote-exec" {
    inline = ["echo 'Droplet is ready'"]
    connection {
      type        = "ssh"
      user        = "root"
      host        = digitalocean_droplet.rsp_image_processing.ipv4_address
      private_key = file(var.ssh_private_key_path)
    }
  }
}

