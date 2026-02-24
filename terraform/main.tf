resource "digitalocean_droplet" "rsp_image_processing" {
  image    = "gpu-l40sx1-48gb"
  name     = "rsp-image-processing"
  region   = "tor1"
  size     = "gpu-h100x1-80gb"
  ssh_keys = [data.digitalocean_ssh_key.terraform.id]
}

