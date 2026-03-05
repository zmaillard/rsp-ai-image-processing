variable "region" {
  type        = string
  description = "The region where the DigitalOcean resources will be created."
  default   = "tor1"
}
variable "size" {
  type        = string
  description = "The size of the Droplets to be created (e.g., 's-1vcpu-1gb')."
  default     = "gpu-h100x1-80gb"
}
variable "digitalocean_token" {
  type        = string
  description = "The DigitalOcean API token to use for authentication."
}

