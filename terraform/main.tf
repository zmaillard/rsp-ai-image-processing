resource "digitalocean_droplet" "rsp_image_processing" {
  image    = data.digitalocean_droplet_snapshot.packer_snapshot.id
  name     = "rsp-image-processing"
  region   = "tor1"
  size     = "gpu-6000adax1-48gb"
  ssh_keys = [data.digitalocean_ssh_key.terraform.id]
}


