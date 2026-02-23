resource "digitalocean_droplet" "rsp_image_processing" {
  image    = "gpu-h100x1-base"
  name     = "rsp-image-processing"
  region   = "nyc2"
  size     = "gpu-h100x1-80gb"
  ssh_keys = [data.digitalocean_ssh_key.terraform.id]
}

