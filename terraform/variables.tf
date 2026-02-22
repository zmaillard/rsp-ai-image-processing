variable "digitalocean_token" {
  type        = string
  description = "The DigitalOcean API token to use for authentication."
}

variable "ssh_private_key_path" {
  type        = string
  description = "The path to the SSH private key for connecting to the droplet."
  default = "~/.ssh/do"
}
