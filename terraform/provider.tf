terraform {
  required_version = ">= 1.0.0"
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
    b2 = {
      source = "Backblaze/b2"
    }
  }
  backend "s3" {
    endpoints = {
      s3 = "https://s3.us-east-005.backblazeb2.com"
    }
    bucket                      = "rsptfstate"
    key                         = "rspimageai.tfstate"
    region                      = "us-east-005"
    skip_credentials_validation = true
    skip_metadata_api_check     = true
    skip_region_validation      = true
    skip_requesting_account_id  = true
    skip_s3_checksum            = true
  }

}

provider "digitalocean" {
  token = var.digitalocean_token
}

provider "b2" {
}
