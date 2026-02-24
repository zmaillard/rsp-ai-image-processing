resource "digitalocean_droplet" "rsp_image_processing" {
  image    = "gpu-h100x1-base"
  name     = "rsp-image-processing"
  region   = "tor1"
  size     = "gpu-l40sx1-48gb"
  ssh_keys = [data.digitalocean_ssh_key.terraform.id]
}

