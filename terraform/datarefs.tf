data "digitalocean_ssh_key" "terraform" {
  name = "Automation"
}

data "digitalocean_droplet_snapshot" "packer_snapshot" {
  name_regex  = "packer-1770866861"
  region      = "tor1"
  most_recent = true
}
