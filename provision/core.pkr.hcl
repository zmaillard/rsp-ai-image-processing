packer {
  required_plugins {
    ansible = {
      version = "~> 1"
      source = "github.com/hashicorp/ansible"
    }

    digitalocean = {
      version = ">= 1.0.4"
      source  = "github.com/digitalocean/digitalocean"
    }
  }
}


variable "digitalocean_token" {
  type = string 
  sensitive = true
}

source "digitalocean" "droplet" {
  api_token = var.digitalocean_token
  image = "gpu-h100x1-base"
  region = "tor1"
  size = "gpu-6000adax1-48gb"
  ssh_username = "root"
}

build {
  sources = ["source.digitalocean.droplet"]

  provisioner "ansible" {
    playbook_file = "./install-hugging-face.yaml"
    extra_arguments = [
      "--vault-password-file", "vaultpassword"
    ]
  }
}
